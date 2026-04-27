#!/usr/bin/env bash
# RTLMap 对比学习训练脚本（边分类 + 图回归 + 超矩形对比损失）
#
# 用法:
#   bash run_contrastive.sh [DATASET_DIR] [DATA_ROOT]
#
# 示例:
#   bash run_contrastive.sh /path/to/dataset /path/to/data_root
#
# 对比学习参数说明:
#   --use-hyperrectangle           启用超矩形对比学习
#   --contrastive-loss-weight     对比损失权重 (默认 0.5)
#   --contrastive-loss-type        损失类型: mse / bce / margin (默认 mse)
#   --contrastive-pairs-per-epoch 每 epoch 采样对比对数量 (默认 512)
#   --contrastive-batch-size      对比 DataLoader batch 大小 (默认 16)
#   --contrastive-margin          margin 模式下不相似对交集上限 (默认 0.2)
#   --hyper-min-margin           超矩形每维度最小宽度 (默认 0.01)

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
    --use-hyperrectangle \
    --contrastive-loss-weight 0.5 \
    --contrastive-loss-type mse \
    --contrastive-pairs-per-epoch 512 \
    --contrastive-batch-size 16 \
    --contrastive-margin 0.2 \
    --hyper-min-margin 0.01 \
    --experiment-name contrastive_hyperrectangle \
    --logger-type tensorboard \
    --checkpoint-dir checkpoints/contrastive \
    --save-top-k 3 \
    --early-stopping-patience 10 \
    --seed 42