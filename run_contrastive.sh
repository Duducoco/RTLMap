#!/usr/bin/env bash
# RTLMap joint contrastive 训练脚本（图回归 + 覆盖向量相似度对比损失）
#
# 用法:
#   bash run_contrastive.sh [DATA_ROOT]
#
# 示例:
#   bash run_contrastive.sh /path/to/data_root
#   DATASET_ROOT=/path/to/coverage-report-extractor/out bash run_contrastive.sh ./data
#
# 首次训练前建议先预热数据和 CodeBERT 缓存:
#   uv run python prepare_contrastive_data.py --data-root ./data_contrastive --encoder-devices 0,1 --text-batch-size 1024 --asm-chunk-files 64
#
# joint loss:
#   L_total = lambda_ce * L_graph + lambda_iou * L_iou + lambda_volume * L_volume
#
# 对比学习参数说明:
#   --joint-contrastive            启用 coverage-vector pair 对比训练
#   --lambda-iou                   逐类型真实体积 IoU 对齐权重
#   --lambda-volume                单样本真实体积校准权重
#   --contrastive-batch-size       对比 DataLoader batch 大小 (默认 16)
#   --pair-candidate-pool-size     每个 anchor 的同模块候选池大小 (默认 128)
#   --lambda-ce                    a/b 两路图回归监督损失权重 (默认 1.0)
#   --volume-warmup-epochs         真实体积权重 warmup epoch 数 (默认 5)
#   --smooth-intersection-temperature 训练阶段平滑交集温度 (默认 0.01)
#   --hyper-min-margin             超矩形每维度最小宽度 (默认 0.01)

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/home/u1/projects/coverage-report-extractor/out}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-2}"
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
    --batch-size 16 \
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
    --joint-contrastive \
    --contrastive-batch-size 16 \
    --pair-candidate-pool-size 128 \
    --pair-relative-low-quota 2 \
    --pair-relative-mid-quota 1 \
    --pair-relative-high-quota 1 \
    --pair-sampling-seed 42 \
    --lambda-ce 1.0 \
    --lambda-iou 1.0 \
    --lambda-volume 0.25 \
    --volume-warmup-epochs 5 \
    --smooth-intersection-temperature 0.01 \
    --hyperrectangle-dim-per-type 5 \
    --hyper-min-margin 0.01 \
    --experiment-name all-4coverage \
    --logger-type tensorboard \
    --checkpoint-dir checkpoints/contrastive \
    --save-top-k 3 \
    --early-stopping-patience 10 \
    --seed 42
