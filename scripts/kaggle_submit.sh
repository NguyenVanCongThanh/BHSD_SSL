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
    echo "  --help, -h                 Xem hướng dẫn"
    echo "=========================================================================="
    exit 0
}

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
fi

PYTHONUNBUFFERED=1 python3 "$KAGGLE_CTL" push "$@"
