#!/bin/bash
# ==============================================================================
# Script tự động submit Notebook Mean Teacher Dual GPU lên Kaggle
# Hỗ trợ Đa Tài Khoản (Multi-Account) & Tự Động Tạo Notebook Mới Chuẩn Hóa
# Sử dụng Kiến Trúc Module Hóa MedSSL-BHSD Kaggle Controller
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"
KAGGLE_CTL="$PROJECT_ROOT/tools/kaggle/kaggle_ctl.py"

ACCOUNT="1"
CLI_USERNAME=""
CLI_KEY=""
CLI_DATASET=""
EXTRA_ARGS=()

show_help() {
    echo "=========================================================================="
    echo "SỬ DỤNG: bash scripts/submit_mean_teacher_kaggle.sh [TÙY CHỌN]"
    echo "=========================================================================="
    echo "  --account, -a <1|2|slug>  Chọn tài khoản để submit (Mặc định: 1 hoặc thanh)"
    echo "  --username, -u <name>     Cập nhật username cho tài khoản"
    echo "  --key, -k <key>           Cập nhật API Key (KGAT_...)"
    echo "  --dataset, -d <slug>      Ghi đè dataset nguồn (Mặc định: thnhnguynvncng/bhsd-fold0)"
    echo "  --test-only               Chỉ kiểm tra kết nối tài khoản mà không submit"
    echo "  --list-accounts           Xem danh sách các tài khoản đang được quản lý"
    echo "  --status                  Kiểm tra trạng thái notebook đang chạy trên Kaggle"
    echo "  --help, -h                Hiển thị hướng dẫn này"
    echo ""
    echo "VÍ DỤ SỬ DỤNG:"
    echo "  1. Submit bằng Tài khoản 1 (Thành - Mặc định):"
    echo "     bash scripts/submit_mean_teacher_kaggle.sh"
    echo ""
    echo "  2. Submit bằng Tài khoản 2 (Bảo Khánh Lê):"
    echo "     bash scripts/submit_mean_teacher_kaggle.sh --account 2"
    echo ""
    echo "  3. Kiểm tra kết nối API của các tài khoản:"
    echo "     bash scripts/submit_mean_teacher_kaggle.sh --test-only"
    echo ""
    echo "  4. Thêm / cập nhật key cho tài khoản 2:"
    echo "     bash scripts/submit_mean_teacher_kaggle.sh -a 2 -u bokhnhl -k KGAT_xxx"
    echo "=========================================================================="
    exit 0
}

# Phân tích tham số
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account|-a)
            ACCOUNT="$2"
            shift 2
            ;;
        --username|-u)
            CLI_USERNAME="$2"
            shift 2
            ;;
        --key|-k)
            CLI_KEY="$2"
            shift 2
            ;;
        --dataset|-d)
            CLI_DATASET="$2"
            shift 2
            ;;
        --list-accounts)
            PYTHONUNBUFFERED=1 python3 "$KAGGLE_CTL" account list
            exit 0
            ;;
        --test-only)
            PYTHONUNBUFFERED=1 python3 "$KAGGLE_CTL" account test
            exit 0
            ;;
        --status)
            PYTHONUNBUFFERED=1 python3 "$KAGGLE_CTL" status --account "$ACCOUNT" --notebook mean_teacher
            exit 0
            ;;
        --help|-h)
            show_help
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Kiểm tra python3 và kaggle CLI
if ! command -v python3 &> /dev/null; then
    echo "[X] Lỗi: Yêu cầu python3 để chạy hệ thống submit!"
    exit 1
fi

if ! command -v kaggle &> /dev/null && [ ! -f "$HOME/.local/bin/kaggle" ]; then
    echo "[*] Không tìm thấy kaggle CLI. Đang tiến hành cài đặt..."
    python3 -m pip install --user kaggle
fi

# Cập nhật username hoặc key nếu người dùng truyền qua CLI
if [ -n "$CLI_USERNAME" ] && [ -n "$CLI_KEY" ]; then
    echo "[*] Đang cập nhật thông tin tài khoản $ACCOUNT vào hệ thống..."
    PYTHONUNBUFFERED=1 python3 "$KAGGLE_CTL" account add \
        --id "$ACCOUNT" \
        --username "$CLI_USERNAME" \
        --key "$CLI_KEY"
fi

# Chuẩn bị tham số gọi kaggle_ctl push
CMD=(python3 "$KAGGLE_CTL" push --account "$ACCOUNT" --notebook mean_teacher)

if [ -n "$CLI_DATASET" ]; then
    CMD+=(--dataset "$CLI_DATASET")
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[*] Khởi chạy submit qua MedSSL-BHSD Kaggle Controller..."
PYTHONUNBUFFERED=1 "${CMD[@]}"
