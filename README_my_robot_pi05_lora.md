# Pi0.5 LoRA Fine-tuning Guide

Pi0.5 (pi05_base) を自前のロボットデータで LoRA fine-tuning するための手順。

## 前提

| 項目 | 値 |
|---|---|
| ベースモデル | pi05_base (PaliGemma 2B + Action Expert 300M) |
| LoRA | gemma_2b_lora (rank=16) + gemma_300m_lora (rank=32) |
| データ形式 | LeRobot v2.1 |
| 学習エンジン | OpenPI (PyTorch DDP) |
| EMA | 無効 (LoRA 必須) |
| freeze_filter | Pi0Config.get_freeze_filter() で自動生成 |

## データ前提

LeRobot v2.1 形式のデータセットが必要:

```
my_lerobot_dataset/
├── meta/
│   ├── info.json         # データセットスキーマ
│   ├── tasks.jsonl       # タスク名一覧
│   └── episodes.jsonl    # エピソードメタデータ
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/               # (オプション) カメラ映像
```

### parquet の必須カラム

| カラム | 型 | 説明 |
|---|---|---|
| `observation.state` | list\<float32\> | ロボット状態 [STATE_DIM] |
| `action` | list\<float32\> | アクション [ACTION_DIM] |
| `episode_index` | int64 | エピソード番号 |
| `frame_index` | int64 | フレーム番号 |
| `timestamp` | float32 | タイムスタンプ |
| `task_index` | int64 | タスクID (tasks.jsonl に対応) |
| `next.done` | bool | エピソード終端フラグ |

## ファイル構成

| ファイル | 役割 |
|---|---|
| `convert_my_data_to_lerobot.py` | 自前データ → LeRobot 変換テンプレート |
| `src/openpi/training/configs/my_robot_pi05_lora.py` | LoRA fine-tune config 定義 |
| `scripts/compute_norm_stats.py` | 正規化統計量の計算 |
| `run_train_my_robot_pi05_lora.sh` | 全パイプラインの実行スクリプト |

## セットアップ

### 1. データ変換

`convert_my_data_to_lerobot.py` の `_load_episode()` 関数を自分のデータ形式に合わせて編集:

```python
def _load_episode(episode_dir: Path) -> dict:
    data = np.load(episode_dir / "episode.npz")
    return {
        "states": data["states"].astype(np.float32),   # [T, STATE_DIM]
        "actions": data["actions"].astype(np.float32),  # [T, ACTION_DIM]
        "task": "pick up the cup",                      # 言語指示
    }
```

次に `STATE_DIM` と `ACTION_DIM` をスクリプト先頭で設定:

```python
STATE_DIM = 7   # 例: 6関節 + 1グリッパー
ACTION_DIM = 7
```

### 2. Config のカスタマイズ

`src/openpi/training/configs/my_robot_pi05_lora.py` を編集:

```python
MY_ACTION_DIM = 7        # あなたのロボットのアクション次元
MY_ACTION_HORIZON = 10   # アクションチャンク長
```

`MyRobotDataConfig.create()` の `RepackTransform` でカラム名をマッピング:

```python
_transforms.RepackTransform({
    "observation/state": "observation.state",  # parquetのカラム名
    "actions": "action",
    "prompt": "prompt",
    # カメラがある場合:
    # "observation/image": "observation.image.top",
})
```

### 3. 学習対象パラメータ

| コンポーネント | パラメータ数 | 学習 |
|---|---|---|
| SigLIP (Vision) | ~400M | frozen |
| PaliGemma (2B LLM) | ~2B | frozen (LoRA rank=16 のみ学習) |
| Action Expert (300M) | ~300M | frozen (LoRA rank=32 のみ学習) |
| Projection layers | ~4M | frozen |
| **LoRA adapters** | **~5M** | **trainable** |

freeze_filter は `Pi0Config.get_freeze_filter()` で自動生成。
LoRA variant を指定すると、対応するベースウェイトが自動的にフリーズされ、
LoRA パラメータのみが学習対象になる。

## 実行方法

### クイックスタート (smoke test)

```bash
# ダミーデータで全パイプラインをテスト (100 steps, batch=4)
bash run_train_my_robot_pi05_lora.sh smoke
```

### 個別ステップ

```bash
# Step 1: データ変換
bash run_train_my_robot_pi05_lora.sh convert

# Step 2: 正規化統計量を計算
bash run_train_my_robot_pi05_lora.sh norm

# Step 3: 学習
bash run_train_my_robot_pi05_lora.sh train
```

### 本番学習 (マルチGPU)

```bash
NUM_GPUS=8 \
DATA_ROOT=/mnt/data \
DATASET_DIR=/mnt/data/my_org/my_robot_data \
REPO_ID=my_org/my_robot_data \
EXP_NAME=lora_prod_v1 \
bash run_train_my_robot_pi05_lora.sh train
```

### 直接コマンド

```bash
# Norm stats
python scripts/compute_norm_stats.py \
    --dataset-dir ./my_lerobot_dataset \
    --output-dir ./assets/my_robot_pi05_lora/my_org/my_robot_data

# Training (single GPU)
python -m openpi.training.train \
    --config-name my_robot_pi05_lora \
    --exp-name lora_v1 \
    --data-root /path/to/datasets \
    --no-wandb

# Training (8 GPU)
torchrun --nproc_per_node=8 -m openpi.training.train \
    --config-name my_robot_pi05_lora \
    --exp-name lora_v1 \
    --data-root /path/to/datasets
```

## Config 一覧

| Config 名 | 用途 | Steps | Batch |
|---|---|---|---|
| `my_robot_pi05_lora_smoke` | Smoke test | 100 | 4 |
| `my_robot_pi05_lora` | Production | 5,000 | 16 |

## チェックポイント

学習済みチェックポイントは以下に保存:

```
./checkpoints/my_robot_pi05_lora/<exp_name>/
├── <step>/
│   ├── model.safetensors   # 全パラメータ (ベース + LoRA)
│   └── optimizer.pt        # オプティマイザ状態
└── latest                  # 最新ステップ番号
```

## トラブルシューティング

### "Normalization stats not found"

```bash
python scripts/compute_norm_stats.py \
    --dataset-dir /path/to/dataset \
    --output-dir ./assets/my_robot_pi05_lora/<repo_id>
```

### "Config not found"

`config.py` に以下が追加されていることを確認:

```python
from openpi.training.configs.my_robot_pi05_lora import MY_ROBOT_PI05_LORA_CONFIGS
_CONFIGS.extend(MY_ROBOT_PI05_LORA_CONFIGS)
```

### VRAM 不足

- `--batch-size 2` でバッチサイズを下げる
- gradient checkpointing を有効化 (train.py に実装済み)
- `--lora-rank 8` でLoRA rank を下げる (PyTorch train.py 使用時)
