#!/usr/bin/env bash
# RTLMap 基础训练脚本（边分类 + 图回归）
#
# 用法:
#   bash run_train.sh [DATASET_DIR] [DATA_ROOT]
#
# 示例:
#   bash run_train.sh /path/to/dataset /path/to/data_root

set -euo pipefail

DATASET_DIR="${1:-/data/rtlmap/dataset}"
DATA_ROOT="${2:-./data}"

uv run python main.py \
    --dataset-dir "$DATASET_DIR" \
    --data-root "$DATA_ROOT" \
    --hidden-dim 256 \
    --num-gnn-layers 4 \
    --dropout 0.1 \
    --max-epochs 100 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --batch-size 16 \
    --num-workers 4 \
    --gradient-clip-val 1.0 \
    --warmup-steps 100 \
    --scheduler cosine \
    --edge-loss-weight 1.0 \
    --graph-loss-weight 1.0 \
    --label-smoothing 0.1 \
    --precision bf16-mixed \
    --use-text-encoder \
    --experiment-name baseline \
    --logger-type tensorboard \
    --checkpoint-dir checkpoints/baseline \
    --save-top-k 3 \
    --early-stopping-patience 10 \
    --seed 42