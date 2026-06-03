#!/usr/bin/env bash
# RTLMap 基础训练脚本（4 个 coverage-report-extractor 数据集，启用 CodeBERT）
#
# 用法:
#   bash run_train.sh [DATA_ROOT]

set -euo pipefail

DATA_ROOT="${1:-./data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/base}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-base}"
    # --dataset-dir /home/u1/projects/coverage-report-extractor/out/archgen_single \
    # --dataset-dir /home/u1/projects/coverage-report-extractor/out/ibex \
    # --dataset-dir /home/u1/projects/coverage-report-extractor/out/picorv32 \
    # --dataset-dir /home/u1/projects/coverage-report-extractor/out/riscv_simple_multicycle \
uv run python main.py \
    --dataset-dir /home/u1/projects/coverage-report-extractor/out/ibex \
    --data-root "$DATA_ROOT" \
    --hidden-dim 256 \
    --num-gnn-layers 4 \
    --dropout 0.1 \
    --max-epochs 100 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --batch-size 1 \
    --accumulate-grad-batches 16 \
    --use-bucketing \
    --token-budget 24576 \
    --num-workers 4 \
    --gradient-clip-val 1.0 \
    --warmup-steps 100 \
    --scheduler cosine \
    --edge-loss-weight 1.0 \
    --graph-loss-weight 1.0 \
    --label-smoothing 0.1 \
    --precision bf16-mixed \
    --use-text-encoder \
    --text-model-name microsoft/codebert-base \
    --text-max-length 512 \
    --text-pooling mean \
    --experiment-name "$EXPERIMENT_NAME" \
    --logger-type tensorboard \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-top-k 3 \
    --early-stopping-patience 10 \
    --seed 42
find smart_run/work -mindepth 2 -maxdepth 2 \( -type d -name coverage -o -name coverage.vdb \) -print0 tar --null -T - -I 'zstd -T0 -19' -cf smart_run/work_coverage_only.tar.zst