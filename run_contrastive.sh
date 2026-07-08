#!/usr/bin/env bash
# RTLMap joint contrastive 训练脚本（图回归 + 覆盖向量相似度对比损失）
#
# 用法:
#   bash run_contrastive.sh [DATASET_DIR] [DATA_ROOT]
#
# 示例:
#   bash run_contrastive.sh /path/to/dataset /path/to/data_root
#
# joint loss:
#   L_total = lambda_ce * (L_graph(a) + L_graph(b)) + lambda_cl * L_contrastive(a, b)
#
# 对比学习参数说明:
#   --joint-contrastive            启用 coverage-vector pair 对比训练
#   --contrastive-loss-type        损失类型: mse / bce / margin (默认 mse)
#   --contrastive-batch-size       对比 DataLoader batch 大小 (默认 16)
#   --contrastive-pairs-per-sample 每个样本最多构造的同模块 pair 数 (默认 4)
#   --contrastive-margin           margin 模式下不相似对交集上限 (默认 0.2)
#   --lambda-ce                    a/b 两路图回归监督损失权重 (默认 1.0)
#   --lambda-cl                    joint 模式下对比损失权重 (默认 0.5)
#   --hyper-min-margin             超矩形每维度最小宽度 (默认 0.01)

set -euo pipefail

DATASET_DIR="${1:-/home/bwq/projects/coverage-report-extractor/out/archgen_ai_single_contrastive}"
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
    --graph-loss-weight 1.0 \
    --precision bf16-mixed \
    --joint-contrastive \
    --contrastive-loss-type mse \
    --contrastive-batch-size 16 \
    --contrastive-pairs-per-sample 4 \
    --contrastive-margin 0.2 \
    --lambda-ce 1.0 \
    --lambda-cl 0.5 \
    --hyper-min-margin 0.01 \
    --experiment-name archgen_ai_single_contrastive \
    --logger-type tensorboard \
    --checkpoint-dir checkpoints/contrastive \
    --save-top-k 3 \
    --early-stopping-patience 10 \
    --seed 42
