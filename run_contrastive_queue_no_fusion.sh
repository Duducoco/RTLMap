#!/usr/bin/env bash
# Run pooled-add joint contrastive experiments for every dataset sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAMES=(archgen_single ibex picorv32 riscv_simple_multicycle)
NO_FUSION_CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/contrastive_no_fusion}"

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
  bash run_contrastive_queue_no_fusion.sh

使用 pooled_add 无融合架构和 joint contrastive 模式依次训练四个数据集:
EOF
    printf '  '
    print_dataset_order
    cat <<'EOF'

每个实验均启用图回归、Hyperrectangle、volume loss 和 IoU loss。

环境变量:
  DATASET_ROOT, ACCELERATOR, DEVICES, CHECKPOINT_DIR

后台运行示例:
  nohup bash run_contrastive_queue_no_fusion.sh > contrastive_queue_no_fusion.log 2>&1 &
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
echo "无融合对比训练顺序: $(print_dataset_order)"
for dataset_name in "${DATASET_NAMES[@]}"; do
    echo "开始无融合对比训练: $dataset_name"
    MODEL_ARCHITECTURE=pooled_add \
        CHECKPOINT_DIR="$NO_FUSION_CHECKPOINT_DIR" \
        bash "$SCRIPT_DIR/run_contrastive.sh" "$dataset_name"
    echo "完成无融合对比训练: $dataset_name"
done

echo "全部无融合对比实验已完成"
