#!/bin/bash
# ==============================================================================
# Script tổng quát submit bất kỳ notebook nào lên Kaggle bằng MedSSL-BHSD Kaggle Controller
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"
KAGGLE_CTL="$PROJECT_ROOT/tools/kaggle/kaggle_ctl.py"

show_help() {
    echo "=========================================================================="
    echo "SỬ DỤNG: bash scripts/kaggle_submit.sh [TÙY CHỌN]"
    echo "=========================================================================="
    echo "  --account, -a <id|slug>    Chọn tài khoản (1: Thành, 2: Khánh, ...)"
    echo "  --notebook, -n <type>      Loại notebook (mean_teacher, pick, ...)"
    echo "  --dataset, -d <slug>       Ghi đè dataset nguồn"
    echo "  --list-accounts            Xem danh sách tài khoản"
    echo "  --test-accounts            Kiểm tra kết nối tất cả tài khoản"
    echo "  --status                   Xem trạng thái notebook"
    echo "  --log                      Kéo riêng file log của notebook về"
    echo "  --help, -h                 Xem hướng dẫn"
    echo "=========================================================================="
    exit 0
}

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
fi

if [ "$1" == "--list-accounts" ]; then
    exec python3 "$KAGGLE_CTL" account list
elif [ "$1" == "--test-accounts" ]; then
    exec python3 "$KAGGLE_CTL" account test
elif [ "$1" == "--status" ] || [ "$1" == "status" ]; then
    shift
    exec python3 "$KAGGLE_CTL" status "$@"
elif [ "$1" == "--log" ] || [ "$1" == "log" ]; then
    shift
    exec python3 "$KAGGLE_CTL" log "$@"
fi

PYTHONUNBUFFERED=1 exec python3 "$KAGGLE_CTL" push "$@"
