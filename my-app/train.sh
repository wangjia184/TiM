#!/usr/bin/env bash
set -euo pipefail

# Launch TiM-style pixel-space training with Accelerate.
#
# Logs:
# - TensorBoard: /workspace/.runs/<YYYYmmdd-HHMMSS>  (created by train.py)
#
# Usage (inside container):
#   bash /workspace/my-app/train.sh
#
# Override any args via env var TRAIN_ARGS, e.g.
#   TRAIN_ARGS="--mixed_precision bf16 --batch_size 128 --patch_size 2" bash /workspace/my-app/train.sh

DEFAULT_ARGS="\
  --mixed_precision bf16 \
  --batch_size 64 \
  --num_workers 4 \
  --epochs 50 \
  --img_size 64 \
  --patch_size 2 \
  --hidden_size 768 \
  --depth 12 \
  --num_heads 12 \
  --sample_every 200 \
  --log_every 20 \
  --use_dir_loss \
  --consistency_ratio 0.1 \
  --diffusion_ratio 0.5 \
  --weight_time_type sqrt \
  --weight_time_tangent \
  --differential_epsilon 0.005 \
  --ema_beta 0.9999 \
"

TRAIN_ARGS="${TRAIN_ARGS:-$DEFAULT_ARGS}"

echo "[train.sh] TRAIN_ARGS=${TRAIN_ARGS}"

# Use 1 process by default (works without an accelerate config file)
accelerate launch --num_processes 1 --mixed_precision bf16 /workspace/my-app/train.py ${TRAIN_ARGS}


