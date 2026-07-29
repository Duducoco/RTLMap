#!/usr/bin/env bash
# Train each dataset sequentially without waiting for or monitoring another process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAMES=(archgen_single ibex picorv32 riscv_simple_multicycle)

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
  bash run_train_queue.sh

不监控其他训练进程，直接依次单独训练四个数据集:
EOF
    printf '  '
    print_dataset_order
    cat <<'EOF'

后台运行示例:
  nohup bash run_train_queue.sh > train_queue.log 2>&1 &
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
echo "训练顺序: $(print_dataset_order)"
for dataset_name in "${DATASET_NAMES[@]}"; do
    echo "开始普通训练: $dataset_name"
    bash "$SCRIPT_DIR/run_train.sh" --dataset "$dataset_name"
    echo "完成普通训练: $dataset_name"
done

echo "全部排队训练已完成"
