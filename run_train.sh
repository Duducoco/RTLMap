#!/usr/bin/env bash
# RTLMap 非对比学习基线训练脚本
#
# 与 run_contrastive.sh 使用相同的数据集、模型和训练配置，仅关闭
# joint contrastive loss 与 hyperrectangle head，用于公平消融对比。
#
# 用法:
#   bash run_train.sh [DATA_ROOT]
#
# 默认复用 run_contrastive.sh 生成的通用图和 CodeBERT 缓存：
#   bash run_train.sh ./data_contrastive

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/u1/projects/coverage-report-extractor/out}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/no_contrastive}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-all-4coverage}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
DATASET_DIRS=(
    "$DATASET_ROOT/archgen_single"
    "$DATASET_ROOT/ibex"
    "$DATASET_ROOT/picorv32"
    "$DATASET_ROOT/riscv_simple_multicycle"
)
DATA_ROOT="${1:-./data_contrastive}"

uv run python main.py \
    --dataset-dir "${DATASET_DIRS[@]}" \
    --data-root "$DATA_ROOT" \
    --hidden-dim 256 \
    --num-gnn-layers 4 \
    --dropout 0.1 \
    --max-epochs 100 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --batch-size 32 \
    --num-workers 4 \
    --gradient-clip-val 1.0 \
    --warmup-steps 100 \
    --scheduler cosine \
    --graph-loss-weight 1.0 \
    --coverage-targets branch line toggle condition \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --precision bf16-mixed \
    --use-text-encoder \
    --text-model-name microsoft/codebert-base \
    --text-output-dim 256 \
    --text-max-length 512 \
    --text-batch-size 1024 \
    --text-pooling mean \
    --experiment-name "$EXPERIMENT_NAME" \
    --logger-type tensorboard \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-top-k 3 \
    --early-stopping-patience 10 \
    --seed 42
