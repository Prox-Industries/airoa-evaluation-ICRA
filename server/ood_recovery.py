"""OOD detection + retry-based recovery for HSR gripper.

Behaviour: when the model is ambiguous on a given frame, **discard the
prediction and re-run inference (blocking)** until either the result
clears the ambiguity threshold or ``max_retries`` is exhausted. The
retry exploits flow-matching noise stochasticity — same observation,
different noise draw, different prediction. After the loop, a
``persistent_ood_action`` decides what to return if the model is still
ambiguous (default: keep the last retry result).

Pipeline per call:
  1. Run inference with the user prompt → ``result``.
  2. Run inference with the counterfactual prompt (Pick<->Place swap of
     the same object family) → ``result_cf``.
  3. Score = max (or mean) over the action chunk of
     ``|gripper_orig - gripper_cf|``. If score >= threshold (and the
     gripper crosses the 0.5 hybrid switch in opposite directions), the
     frame is OOD.
  4. If OOD, discard ``result`` and go back to step 1 with a fresh noise
     draw. Repeat up to ``max_retries`` times.
  5. If still OOD after all retries, apply ``persistent_ood_action``.

Cost: 2 inferences per call when OOD is not detected. When OOD fires,
``2 + 2 * retries`` inferences. Disable entirely with
``OOD_ENABLED=false`` (default off in the bare config; the ICRA Docker
image enables it by default via ENV).
"""
from __future__ import annotations

import dataclasses
import logging
from collections import deque
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
DEFAULT_GRIPPER_THRESHOLD = 0.5  # hybrid open/close switch, see hsr_policy.py


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
    enabled: bool = False                # ON by default in Dockerfile; library default OFF.

    # ---- ambiguity detection ----
    ambiguity_threshold: float = 0.1     # |g_orig - g_cf| over chunk; >= triggers
    chunk_aggregation: str = "max"       # "max" | "mean"
    require_threshold_crossing: bool = True

    # ---- retry-on-OOD (blocking loop) ----
    retry_on_ood: bool = True
    max_retries: int = 3                 # blocking attempts before giving up

    # ---- fallback when retries don't clear the OOD ----
    # "keep_last" : return the most recent (still-ambiguous) retry. Trusts the
    #               distribution of retries to be no worse than the original.
    # "pa_rule"   : override gripper to the PA-expected value (close=Pick,
    #               open=Place). Safety-first.
    # "keep_first": kept for forward compat (currently behaves like keep_last
    #               because we rebind ``result`` per retry).
    persistent_ood_action: str = "keep_last"

    # ---- override values when persistent_ood_action == "pa_rule" ----
    gripper_close_value: float = 0.0
    gripper_open_value: float = 1.0

    # ---- hysteresis on persistent OOD ----
    min_consecutive_ood: int = 1

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
            retry_on_ood=_bool("OOD_RETRY_ON_OOD", True),
            max_retries=_int("OOD_MAX_RETRIES", 3),
            persistent_ood_action=_str("OOD_PERSISTENT_ACTION", "keep_last"),
            gripper_close_value=_float("OOD_GRIPPER_CLOSE", 0.0),
            gripper_open_value=_float("OOD_GRIPPER_OPEN", 1.0),
            min_consecutive_ood=_int("OOD_MIN_CONSECUTIVE", 1),
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
    """Wrap a base policy so that ambiguous-frame predictions are blocked,
    discarded, and re-run until the ambiguity score falls below threshold.

    Transparent passthrough when ``config.enabled`` is False.
    """

    def __init__(self, base_policy: _PolicyLike, config: OODRecoveryConfig):
        self._base = base_policy
        self._config = config
        self._ood_history: deque[bool] = deque(maxlen=max(config.min_consecutive_ood, 1))
        self.stats: dict[str, int] = {
            "total_calls": 0,
            "skipped_disabled": 0,
            "skipped_unknown_pa": 0,
            "initial_ood_detected": 0,
            "retries_attempted": 0,
            "retry_cleared_ood": 0,
            "persistent_ood": 0,
            "fallback_applied": 0,
        }
        if config.enabled:
            logger.info(
                "OOD recovery ENABLED: threshold=%.3f agg=%s hyst=%d retry=%s "
                "max_retries=%d persistent_action=%s strict_pa=%s",
                config.ambiguity_threshold, config.chunk_aggregation,
                config.min_consecutive_ood, config.retry_on_ood,
                config.max_retries, config.persistent_ood_action,
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
        }
        return md

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

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.stats["total_calls"] += 1

        if not self._config.enabled:
            self.stats["skipped_disabled"] += 1
            return self._base.infer(obs)

        prompt = obs.get("prompt") if isinstance(obs, dict) else None
        cf_prompt = swap_pa_verb(prompt, strict=self._config.pa_classifier_strict)
        pa_verb = classify_pa(prompt)

        # Initial inference.
        result = self._base.infer(obs)

        # Unknown PA → cannot construct a counterfactual → skip the OOD logic.
        if cf_prompt is None or pa_verb not in ("pick", "place"):
            self.stats["skipped_unknown_pa"] += 1
            self._ood_history.clear()
            result["ood_recovery"] = {"applied": False, "reason": "unknown_pa"}
            return result

        # Detect OOD on the initial inference.
        is_ood, score, crossing = self._check_ood(result, cf_prompt, obs)
        attempt_log = [{"attempt": 0, "score": score, "crossing": crossing,
                        "is_ood": is_ood}]

        # Blocking retry loop: discard prediction and re-infer until clean
        # (or budget exhausted).
        retries_done = 0
        if is_ood and self._config.retry_on_ood:
            self.stats["initial_ood_detected"] += 1
            for retry_i in range(self._config.max_retries):
                self.stats["retries_attempted"] += 1
                retries_done += 1
                # Discard the previous prediction; re-run with fresh noise.
                result = self._base.infer(obs)
                is_ood, score, crossing = self._check_ood(result, cf_prompt, obs)
                attempt_log.append({
                    "attempt": retry_i + 1, "score": score,
                    "crossing": crossing, "is_ood": is_ood,
                })
                if not is_ood:
                    self.stats["retry_cleared_ood"] += 1
                    break
            if self._config.log_overrides:
                logger.info(
                    "OOD retry: pa=%s attempts=%d final_ood=%s final_score=%.4f",
                    pa_verb, retries_done, is_ood, score,
                )

        # Hysteresis on the post-retry status.
        self._ood_history.append(is_ood)
        fire = (
            is_ood
            and len(self._ood_history) >= self._config.min_consecutive_ood
            and all(self._ood_history)
        )

        applied = False
        applied_action = None
        if fire:
            self.stats["persistent_ood"] += 1
            actions = np.asarray(result["actions"], dtype=np.float32)
            if self._config.persistent_ood_action == "pa_rule":
                override_value = (
                    self._config.gripper_close_value if pa_verb == "pick"
                    else self._config.gripper_open_value
                )
                actions[:, GRIPPER_DIM] = override_value
                result["actions"] = actions
                applied = True
                applied_action = "pa_rule"
                self.stats["fallback_applied"] += 1
                if self._config.log_overrides:
                    logger.info(
                        "OOD persistent fallback (pa_rule): pa=%s score=%.4f "
                        "set gripper=%g", pa_verb, score, override_value,
                    )
            elif self._config.persistent_ood_action == "keep_last":
                applied_action = "keep_last"
            elif self._config.persistent_ood_action == "keep_first":
                # Forward-compat alias; behaves like keep_last in this loop.
                applied_action = "keep_first"
            else:
                logger.warning(
                    "Unknown persistent_ood_action=%r; treating as keep_last",
                    self._config.persistent_ood_action,
                )
                applied_action = "keep_last"

        if self._config.log_each_call:
            logger.info(
                "OOD call: pa=%s retries=%d final_ood=%s score=%.4f fire=%s",
                pa_verb, retries_done, is_ood, score, fire,
            )

        result["ood_recovery"] = {
            "applied": applied,
            "applied_action": applied_action,
            "pa_verb": pa_verb,
            "ambiguity_score": score,
            "crossing": crossing,
            "is_ambiguous": is_ood,
            "retries": retries_done,
            "attempts": attempt_log,
            "history_len": len(self._ood_history),
        }
        return result
