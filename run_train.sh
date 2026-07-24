#!/usr/bin/env bash
# RTLMap 仅监督训练脚本
#
# 与 run_contrastive.sh 使用相同的数据集、模型和训练配置，仅关闭
# joint contrastive loss 与 hyperrectangle head，用于公平消融对比。
#
# 运行 `bash run_train.sh --help` 查看命令行选项和示例。

set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  bash run_train.sh [DATA_ROOT]
  bash run_train.sh --dataset NAME [--data-root PATH]

无参数时合并训练全部四个数据集；使用 --dataset 时单独训练指定数据集。

选项:
  --dataset NAME    单独训练指定数据集
  --data-root PATH  指定 processed 数据缓存目录
  -h, --help        显示帮助

示例:
  bash run_train.sh
  bash run_train.sh ./data_contrastive
  bash run_train.sh --dataset ibex

环境变量:
  DATASET_ROOT, ACCELERATOR, DEVICES, CHECKPOINT_DIR, EXPERIMENT_NAME
  MODEL_ARCHITECTURE  模型架构: perceiver_fusion（默认）、pooled_add 或 rtl_gcn
EOF
}

DATASET_ROOT="${DATASET_ROOT:-/home/u1/projects/coverage-report-extractor/out}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/no_contrastive}"
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-perceiver_fusion}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "$MODEL_ARCHITECTURE" in
    perceiver_fusion|pooled_add|rtl_gcn)
        ;;
    *)
        echo "错误: MODEL_ARCHITECTURE 必须是 perceiver_fusion、pooled_add 或 rtl_gcn" >&2
        exit 2
        ;;
esac

DATASET_NAME=""
DATA_ROOT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            if [[ $# -lt 2 ]]; then
                echo "错误: --dataset 需要一个数据集名称" >&2
                exit 2
            fi
            if [[ -n "$DATASET_NAME" ]]; then
                echo "错误: 只能选择一个数据集" >&2
                exit 2
            fi
            DATASET_NAME="$2"
            shift 2
            ;;
        --data-root)
            if [[ $# -lt 2 ]]; then
                echo "错误: --data-root 需要一个目录路径" >&2
                exit 2
            fi
            if [[ -n "$DATA_ROOT" ]]; then
                echo "错误: 只能指定一个 data root" >&2
                exit 2
            fi
            DATA_ROOT="$2"
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
            if [[ -n "$DATA_ROOT" ]]; then
                echo "错误: 只能指定一个 data root" >&2
                exit 2
            fi
            DATA_ROOT="$1"
            shift
            ;;
    esac
done

if [[ -n "$DATASET_NAME" ]]; then
    if [[ ! "$DATASET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "$DATASET_NAME" == "." ]] || [[ "$DATASET_NAME" == ".." ]]; then
        echo "错误: 无效的数据集名称: $DATASET_NAME" >&2
        exit 2
    fi

    DATASET_DIRS=("$DATASET_ROOT/$DATASET_NAME")
    DATA_ROOT="${DATA_ROOT:-./data_contrastive_${DATASET_NAME}}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATASET_NAME}-4coverage}"
else
    DATASET_DIRS=(
        "$DATASET_ROOT/archgen_single"
        "$DATASET_ROOT/ibex"
        "$DATASET_ROOT/picorv32"
        "$DATASET_ROOT/riscv_simple_multicycle"
    )
    DATA_ROOT="${DATA_ROOT:-./data_contrastive}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-all-4coverage}"
fi

for dataset_dir in "${DATASET_DIRS[@]}"; do
    if [[ ! -f "$dataset_dir/manifest.json" ]]; then
        echo "错误: 数据集 manifest 不存在: $dataset_dir/manifest.json" >&2
        exit 2
    fi
done

uv run python main.py \
    --dataset-dir "${DATASET_DIRS[@]}" \
    --data-root "$DATA_ROOT" \
    --model-architecture "$MODEL_ARCHITECTURE" \
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
