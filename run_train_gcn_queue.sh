#!/usr/bin/env bash
# Train every dataset sequentially with a standard GCN on the RTL branch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAMES=(archgen_single ibex picorv32 riscv_simple_multicycle)
GCN_CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/rtl_gcn}"

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
  bash run_train_gcn_queue.sh

使用 rtl_gcn 架构依次单独训练四个数据集。仅 RTL 消息传递替换为标准 GCN，
ASM 编码、Perceiver 融合、池化和任务 head 保持不变:
EOF
    printf '  '
    print_dataset_order
    cat <<'EOF'

环境变量:
  DATASET_ROOT, ACCELERATOR, DEVICES, CHECKPOINT_DIR

后台运行示例:
  nohup bash run_train_gcn_queue.sh > train_gcn_queue.log 2>&1 &
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
echo "RTL GCN 训练顺序: $(print_dataset_order)"
for dataset_name in "${DATASET_NAMES[@]}"; do
    experiment_name="${dataset_name}-4coverage-rtl-gcn"
    echo "开始 RTL GCN 训练: $dataset_name"
    MODEL_ARCHITECTURE=rtl_gcn \
        EXPERIMENT_NAME="$experiment_name" \
        CHECKPOINT_DIR="$GCN_CHECKPOINT_DIR" \
        bash "$SCRIPT_DIR/run_train.sh" --dataset "$dataset_name"
    echo "完成 RTL GCN 训练: $dataset_name"
done

echo "全部 RTL GCN 排队训练已完成"
