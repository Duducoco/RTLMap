#!/usr/bin/env bash
# RTLMap joint contrastive 训练脚本（图回归 + 覆盖向量相似度对比损失）
#
# 运行 `bash run_contrastive.sh --help` 查看命令行选项和示例。
#
# joint loss:
#   L_total = lambda_ce * L_graph + lambda_iou * L_iou + lambda_volume * L_volume
#
# 对比学习参数说明:
#   --joint-contrastive            启用 coverage-vector pair 对比训练
#   --lambda-iou                   逐类型真实体积 IoU 对齐权重
#   --lambda-volume                单样本真实体积校准权重
#   --contrastive-batch-size       对比 DataLoader batch 大小 (默认 16)
#   --pair-candidate-pool-size     每个 anchor 的同模块候选池大小 (默认 256)
#   --lambda-ce                    a/b 两路图回归监督损失权重 (默认 1.0)
#   --volume-warmup-epochs         真实体积权重 warmup epoch 数 (默认 5)
#   --smooth-intersection-temperature 训练阶段平滑交集温度 (默认 0.01)
#   --hyper-min-margin             超矩形每维度最小宽度 (默认 0.01)

set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  bash run_contrastive.sh [DATASET_NAME] [选项]

选项:
  --dataset NAME    用选项形式选择数据集 (默认: ibex)
  --data-root PATH  指定 processed 数据缓存目录
  --ckpt-path PATH  从指定 checkpoint 恢复训练
  -h, --help        显示帮助

示例:
  bash run_contrastive.sh archgen_single
  bash run_contrastive.sh ibex
  bash run_contrastive.sh picorv32
  bash run_contrastive.sh riscv_simple_multicycle

环境变量:
  DATASET_ROOT, CKPT_PATH, ACCELERATOR, DEVICES
EOF
}

DATASET_ROOT="${DATASET_ROOT:-/home/u1/projects/coverage-report-extractor/out}"
DATASET_NAME="ibex"
DATA_ROOT=""
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-2}"
CKPT_PATH="${CKPT_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_SELECTED=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            if [[ $# -lt 2 ]]; then
                echo "错误: --dataset 需要一个数据集名称" >&2
                exit 2
            fi
            if [[ "$DATASET_SELECTED" == true ]]; then
                echo "错误: 只能选择一个数据集" >&2
                exit 2
            fi
            DATASET_NAME="$2"
            DATASET_SELECTED=true
            shift 2
            ;;
        --data-root)
            if [[ $# -lt 2 ]]; then
                echo "错误: --data-root 需要一个目录路径" >&2
                exit 2
            fi
            DATA_ROOT="$2"
            shift 2
            ;;
        --ckpt-path)
            if [[ $# -lt 2 ]]; then
                echo "错误: --ckpt-path 需要一个 checkpoint 路径" >&2
                exit 2
            fi
            CKPT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "错误: 未知参数 $1" >&2
            exit 2
            ;;
        *)
            if [[ "$DATASET_SELECTED" == true ]]; then
                echo "错误: 只能选择一个数据集" >&2
                exit 2
            fi
            DATASET_NAME="$1"
            DATASET_SELECTED=true
            shift
            ;;
    esac
done

if [[ ! "$DATASET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$DATASET_NAME" == "." ]] || [[ "$DATASET_NAME" == ".." ]]; then
    echo "错误: 无效的数据集名称: $DATASET_NAME" >&2
    exit 2
fi

DATASET_DIR="$DATASET_ROOT/$DATASET_NAME"
if [[ ! -f "$DATASET_DIR/manifest.json" ]]; then
    echo "错误: 数据集 manifest 不存在: $DATASET_DIR/manifest.json" >&2
    exit 2
fi

DATASET_DIRS=("$DATASET_DIR")
DATA_ROOT="${DATA_ROOT:-./data_contrastive_${DATASET_NAME}}"
EXPERIMENT_NAME="${DATASET_NAME}-4coverage-split-rect-mlp-asm-readout"

set --
if [[ -n "$CKPT_PATH" ]]; then
    set -- --ckpt-path "$CKPT_PATH"
fi

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
    --scheduler plateau \
    --plateau-factor 0.5 \
    --plateau-patience 5 \
    --min-lr 1e-6 \
    --graph-loss-weight 1.0 \
    --graph-relative-loss-weight 0.1 \
    --graph-relative-loss-floor 0.1 \
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
    --pair-candidate-pool-size 256 \
    --pair-relative-low-quota 2 \
    --pair-relative-mid-quota 1 \
    --pair-relative-high-quota 1 \
    --pair-sampling-seed 42 \
    --lambda-ce 1.0 \
    --lambda-iou 1.0 \
    --lambda-volume 1.0 \
    --volume-warmup-epochs 0 \
    --smooth-intersection-temperature 0.01 \
    --hyperrectangle-dim-per-type 5 \
    --hyper-min-margin 0.01 \
    --experiment-name "$EXPERIMENT_NAME" \
    --logger-type tensorboard \
    --checkpoint-dir checkpoints/contrastive \
    --save-top-k 3 \
    --early-stopping-patience 30 \
    --seed 42 \
    "$@"
