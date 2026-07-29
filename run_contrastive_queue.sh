#!/usr/bin/env bash
# Wait for an existing contrastive run, then train the remaining datasets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAMES=(archgen_single riscv_simple_multicycle ibex)

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
  bash run_contrastive_queue.sh [选项]

无参数时自动检测当前 run_contrastive.sh 或 joint-contrastive main.py 进程。
当前训练结束后依次运行:
EOF
    printf '  '
    print_dataset_order
    cat <<'EOF'
选项:
  --wait-pid PID             显式指定需要等待的当前训练进程
  --poll-interval SECONDS    检查间隔秒数 (默认: 30)
  -h, --help                 显示帮助

后台运行示例:
  nohup bash run_contrastive_queue.sh > contrastive_queue.log 2>&1 &
EOF
}

WAIT_PID=""
POLL_INTERVAL="${POLL_INTERVAL_SECONDS:-30}"

detect_training_pids() {
    local candidate_pids
    local pid
    local process_cwd

    candidate_pids="$(
        {
            pgrep -f '(^|[[:space:]/])run_contrastive\.sh([[:space:]]|$)' || true
            pgrep -f 'main\.py .*--joint-contrastive' || true
        } | sort -u
    )"
    for pid in $candidate_pids; do
        process_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
        if [[ "$process_cwd" == "$SCRIPT_DIR" ]]; then
            printf '%s\n' "$pid"
        fi
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait-pid)
            if [[ $# -lt 2 ]]; then
                echo "错误: --wait-pid 需要一个 PID" >&2
                exit 2
            fi
            WAIT_PID="$2"
            shift 2
            ;;
        --poll-interval)
            if [[ $# -lt 2 ]]; then
                echo "错误: --poll-interval 需要秒数" >&2
                exit 2
            fi
            POLL_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1" >&2
            exit 2
            ;;
    esac
done

if [[ -n "$WAIT_PID" ]] && [[ ! "$WAIT_PID" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误: --wait-pid 必须是正整数" >&2
    exit 2
fi
if [[ ! "$POLL_INTERVAL" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || [[ "$POLL_INTERVAL" =~ ^0*([.]0*)?$ ]]; then
    echo "错误: --poll-interval 必须是正数" >&2
    exit 2
fi

WAIT_PIDS="${WAIT_PID:-$(detect_training_pids)}"
if [[ -z "$WAIT_PIDS" ]]; then
    echo "错误: 未检测到正在运行的对比训练，请通过 --wait-pid 指定 PID" >&2
    exit 2
fi

echo "等待当前训练进程结束: PID ${WAIT_PIDS//$'\n'/, }"
while :; do
    training_is_running=false
    for pid in $WAIT_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            training_is_running=true
            break
        fi
    done
    if [[ "$training_is_running" == false ]]; then
        break
    fi
    sleep "$POLL_INTERVAL"
done

cd "$SCRIPT_DIR"
for dataset_name in "${DATASET_NAMES[@]}"; do
    echo "开始对比训练: $dataset_name"
    bash "$SCRIPT_DIR/run_contrastive.sh" "$dataset_name"
    echo "完成对比训练: $dataset_name"
done

echo "全部排队训练已完成"
