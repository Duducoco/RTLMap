#!/usr/bin/env bash
# Train each dataset sequentially with independent GNN branches and pooled addition.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAMES=(archgen_single ibex picorv32 riscv_simple_multicycle)
NO_FUSION_CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/no_fusion}"

print_dataset_order() {
    local dataset_name
    local separator=""

    for dataset_name in "${DATASET_NAMES[@]}"; do
        printf '%s%s' "$separator" "$dataset_name"
        separator=" -> "
    done
    printf '\n'
}

usage() {
    cat <<'EOF'
用法:
  bash run_train_queue_no_fusion.sh

使用 pooled_add 架构依次单独训练四个数据集，不创建或执行 Perceiver 特征融合:
EOF
    printf '  '
    print_dataset_order
    cat <<'EOF'

环境变量:
  DATASET_ROOT, ACCELERATOR, DEVICES, CHECKPOINT_DIR

后台运行示例:
  nohup bash run_train_queue_no_fusion.sh > train_queue_no_fusion.log 2>&1 &
EOF
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1" >&2
            exit 2
            ;;
    esac
fi

cd "$SCRIPT_DIR"
echo "无融合训练顺序: $(print_dataset_order)"
for dataset_name in "${DATASET_NAMES[@]}"; do
    experiment_name="${dataset_name}-4coverage-no-fusion"
    echo "开始无融合训练: $dataset_name"
    MODEL_ARCHITECTURE=pooled_add \
        EXPERIMENT_NAME="$experiment_name" \
        CHECKPOINT_DIR="$NO_FUSION_CHECKPOINT_DIR" \
        bash "$SCRIPT_DIR/run_train.sh" --dataset "$dataset_name"
    echo "完成无融合训练: $dataset_name"
done

echo "全部无融合排队训练已完成"
