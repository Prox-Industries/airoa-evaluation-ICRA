"""OOD detection + frame-by-frame hold recovery for HSR gripper ambiguity.

Design (stateful, across infer calls — no in-call retry loop):

  for each ``infer(obs_t)`` from the WebSocket client:
    1. Run inference with the user prompt → ``result`` (chunk).
    2. Run inference with the counterfactual prompt (Pick<->Place swap of
       the same object family) → ``result_cf``.
    3. Score = max (or mean) over the action chunk of
       ``|gripper_orig - gripper_cf|``. The frame is OOD when the score
       crosses ``ambiguity_threshold`` (and the gripper crosses the 0.5
       open/close switch in opposite directions, if required).
    4. If NOT OOD → reset the consecutive-ood counter, return the chunk.
    5. If OOD → increment the counter and:
         * counter <= max_holds  →  return a *hold* chunk that keeps the
                                    robot in place for one client tick
                                    (~100 ms). The next ``infer`` call
                                    will receive a *fresh* obs from the
                                    client, so retries naturally use new
                                    observations rather than re-running
                                    on the same one.
         * counter >  max_holds  →  budget exhausted; reset the counter,
                                    return the original chunk anyway.
                                    OOD detection resumes immediately on
                                    the next call.
    6. The counter is also reset on prompt change (episode boundary).

The hold chunk is *gripper-safe*. Naïve zero would be interpreted by the
client as "close gripper" because the discrete/hybrid threshold is 0.5
(see ``deploy/hsr_policy_client/scripts/hsr_policy.py``). To keep the
gripper at its current state (and avoid crushing held objects) we copy
``obs["state"][5]`` into ``action[:, 5]``; the rest of the chunk is zero
(arm/head deltas → joint hold, base velocity → no movement).

Inference cost: 2 base inferences per call (orig + cf). No in-call
retries. The wrapper is a transparent passthrough when
``OOD_ENABLED=false`` (the library default).
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class _PolicyLike(Protocol):
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]: ...
    metadata: dict[str, Any]


# 11-D HSR raw action layout (post _decode_actions_inv):
#   [arm x5, gripper x1, head x2, base_x, base_y, base_t]
GRIPPER_DIM = 5
DEFAULT_GRIPPER_THRESHOLD = 0.5  # client-side hybrid open/close switch
ACTION_DIM = 11
STATE_GRIPPER_DIM = 5            # obs["state"] is the 8-D HSR state vector


# 6 known PAs (verified across all 3,971 task6911 parquets — exhaustive set
# the baseline was trained on). Each entry pairs Pick <-> Place of the same
# object family, so we can synthesise a counterfactual prompt locally.
PA_PAIRS: list[tuple[str, str]] = [
    ("Pick up the coffee bottle on the right",
     "Place the coffee bottle into the box labeled 1"),
    ("Pick up the box labeled 2",
     "Place the box next to the box labeled 1"),
    ("Pick up the mug that is not at a rectangle corner.",
     "Place the mug at the missing rectangle corner."),
]


@dataclasses.dataclass
class OODRecoveryConfig:
    """Runtime config — every field has an OOD_<UPPER> env-var override."""

    # ---- master switch ----
    enabled: bool = False

    # ---- ambiguity detection ----
    ambiguity_threshold: float = 0.1
    chunk_aggregation: str = "max"             # "max" | "mean"
    require_threshold_crossing: bool = True

    # ---- frame-by-frame hold budget ----
    max_holds: int = 3                         # consecutive holds before forced release

    # ---- PA classifier behaviour ----
    pa_classifier_strict: bool = True

    # ---- logging ----
    log_each_call: bool = False
    log_overrides: bool = True

    @classmethod
    def from_env(cls, env: dict | None = None) -> "OODRecoveryConfig":
        import os
        env = env if env is not None else os.environ

        def _bool(key: str, default: bool) -> bool:
            v = env.get(key)
            if v is None:
                return default
            return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}

        def _float(key: str, default: float) -> float:
            v = env.get(key)
            return float(v) if v is not None else default

        def _int(key: str, default: int) -> int:
            v = env.get(key)
            return int(v) if v is not None else default

        def _str(key: str, default: str) -> str:
            return env.get(key, default)

        return cls(
            enabled=_bool("OOD_ENABLED", False),
            ambiguity_threshold=_float("OOD_AMBIGUITY_THRESHOLD", 0.1),
            chunk_aggregation=_str("OOD_CHUNK_AGGREGATION", "max"),
            require_threshold_crossing=_bool("OOD_REQUIRE_CROSSING", True),
            max_holds=_int("OOD_MAX_HOLDS", 3),
            pa_classifier_strict=_bool("OOD_PA_STRICT", True),
            log_each_call=_bool("OOD_LOG_EACH_CALL", False),
            log_overrides=_bool("OOD_LOG_OVERRIDES", True),
        )


def classify_pa(prompt: str | None) -> str | None:
    if not prompt:
        return None
    p = prompt.strip().lower()
    if p.startswith("pick"):
        return "pick"
    if p.startswith("place"):
        return "place"
    return None


def swap_pa_verb(prompt: str | None, *, strict: bool = True) -> str | None:
    if prompt is None:
        return None
    pr = prompt.strip()
    for pick, place in PA_PAIRS:
        if pr == pick:
            return place
        if pr == place:
            return pick
    if strict:
        return None
    pl = pr.lower()
    if pl.startswith("pick up"):
        return "Place" + pr[len("Pick up"):]
    if pl.startswith("place"):
        return "Pick up" + pr[len("Place"):]
    return None


class OODRecoveryPolicy:
    """Stateful wrapper that emits a gripper-safe hold chunk on ambiguous
    frames and forces a release after ``max_holds`` consecutive holds.

    Transparent passthrough when ``config.enabled`` is False.
    """

    def __init__(self, base_policy: _PolicyLike, config: OODRecoveryConfig):
        self._base = base_policy
        self._config = config
        self._consecutive_ood_count: int = 0
        self._last_prompt: str | None = None
        self.stats: dict[str, int] = {
            "total_calls": 0,
            "skipped_disabled": 0,
            "skipped_unknown_pa": 0,
            "ood_detected": 0,
            "holds_emitted": 0,
            "released_after_max_holds": 0,
            "episode_resets": 0,
        }
        if config.enabled:
            logger.info(
                "OOD recovery ENABLED (frame-by-frame hold mode): "
                "threshold=%.3f agg=%s max_holds=%d crossing_required=%s strict_pa=%s",
                config.ambiguity_threshold, config.chunk_aggregation,
                config.max_holds, config.require_threshold_crossing,
                config.pa_classifier_strict,
            )
        else:
            logger.info("OOD recovery disabled (transparent passthrough)")

    @property
    def metadata(self) -> dict[str, Any]:
        md = dict(getattr(self._base, "metadata", {}) or {})
        md["ood_recovery"] = {
            "enabled": self._config.enabled,
            "config": dataclasses.asdict(self._config),
            "mode": "frame_hold",
        }
        return md

    # ----------------------------------------------------------------
    def _check_ood(
        self, result: dict, cf_prompt: str, obs: dict
    ) -> tuple[bool, float, bool]:
        """Run a counterfactual inference and decide whether ``result`` is OOD."""
        obs_cf = dict(obs)
        obs_cf["prompt"] = cf_prompt
        result_cf = self._base.infer(obs_cf)

        actions = np.asarray(result["actions"], dtype=np.float32)
        actions_cf = np.asarray(result_cf["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] <= GRIPPER_DIM:
            return False, 0.0, False

        g_orig = actions[:, GRIPPER_DIM]
        g_cf = actions_cf[:, GRIPPER_DIM]
        diff = np.abs(g_orig - g_cf)
        score = float(diff.mean()) if self._config.chunk_aggregation == "mean" \
                else float(diff.max())
        crossing = bool(
            ((g_orig > DEFAULT_GRIPPER_THRESHOLD) !=
             (g_cf > DEFAULT_GRIPPER_THRESHOLD)).any()
        )
        is_amb = score >= self._config.ambiguity_threshold
        if self._config.require_threshold_crossing:
            is_amb = is_amb and crossing
        return is_amb, score, crossing

    def _make_hold_action(self, T: int, obs: dict[str, Any]) -> np.ndarray:
        """Gripper-safe hold chunk.

        arm/head are delta-form (zero = joint hold), gripper is absolute
        position (must keep the current value to avoid an unintended
        close), base is velocity (zero = no movement).
        """
        hold = np.zeros((max(T, 1), ACTION_DIM), dtype=np.float32)
        state = np.asarray(obs.get("state", np.zeros(8)), dtype=np.float32) \
                if isinstance(obs, dict) else np.zeros(8, dtype=np.float32)
        if state.ndim == 1 and state.shape[0] > STATE_GRIPPER_DIM:
            hold[:, GRIPPER_DIM] = float(state[STATE_GRIPPER_DIM])
        return hold

    # ----------------------------------------------------------------
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.stats["total_calls"] += 1

        if not self._config.enabled:
            self.stats["skipped_disabled"] += 1
            return self._base.infer(obs)

        prompt = obs.get("prompt") if isinstance(obs, dict) else None

        # Episode boundary: prompt change → reset the consecutive counter.
        if prompt != self._last_prompt:
            if self._last_prompt is not None:
                self.stats["episode_resets"] += 1
            self._consecutive_ood_count = 0
            self._last_prompt = prompt

        cf_prompt = swap_pa_verb(prompt, strict=self._config.pa_classifier_strict)
        pa_verb = classify_pa(prompt)

        # Always run the user-prompt inference first.
        result = self._base.infer(obs)

        # Unknown PA → cannot construct a counterfactual → skip OOD logic.
        if cf_prompt is None or pa_verb not in ("pick", "place"):
            self.stats["skipped_unknown_pa"] += 1
            self._consecutive_ood_count = 0
            result["ood_recovery"] = {
                "applied": False, "reason": "unknown_pa",
                "consecutive_ood": 0,
            }
            return result

        is_ood, score, crossing = self._check_ood(result, cf_prompt, obs)

        if not is_ood:
            # Clear frame: reset counter, return chunk as-is.
            self._consecutive_ood_count = 0
            result["ood_recovery"] = {
                "applied": False,
                "is_ambiguous": False,
                "ambiguity_score": score,
                "crossing": crossing,
                "consecutive_ood": 0,
                "pa_verb": pa_verb,
            }
            if self._config.log_each_call:
                logger.info(
                    "OOD clear: pa=%s score=%.4f crossing=%s",
                    pa_verb, score, crossing,
                )
            return result

        # OOD detected.
        self.stats["ood_detected"] += 1
        self._consecutive_ood_count += 1

        if self._consecutive_ood_count <= self._config.max_holds:
            # Within the hold budget: emit a gripper-safe hold chunk.
            self.stats["holds_emitted"] += 1
            actions_orig = np.asarray(result["actions"])
            T = actions_orig.shape[0] if actions_orig.ndim == 2 else 1
            hold = self._make_hold_action(T, obs)
            result["actions"] = hold
            result["ood_recovery"] = {
                "applied": True,
                "action": "hold",
                "consecutive_ood": self._consecutive_ood_count,
                "max_holds": self._config.max_holds,
                "ambiguity_score": score,
                "crossing": crossing,
                "pa_verb": pa_verb,
            }
            if self._config.log_overrides:
                logger.info(
                    "OOD hold: pa=%s consecutive=%d/%d score=%.4f "
                    "→ zero arm/head/base + gripper=%.4f (state-keep)",
                    pa_verb, self._consecutive_ood_count, self._config.max_holds,
                    score, float(hold[0, GRIPPER_DIM]),
                )
            return result

        # Budget exhausted: release, reset counter, OOD detection resumes
        # immediately on the next call.
        self.stats["released_after_max_holds"] += 1
        self._consecutive_ood_count = 0
        result["ood_recovery"] = {
            "applied": False,
            "action": "released_after_max_holds",
            "ambiguity_score": score,
            "crossing": crossing,
            "pa_verb": pa_verb,
            "max_holds": self._config.max_holds,
        }
        if self._config.log_overrides:
            logger.info(
                "OOD release: pa=%s held %d frames, releasing original chunk; "
                "counter reset, detection resumes next call",
                pa_verb, self._config.max_holds,
            )
        return result
