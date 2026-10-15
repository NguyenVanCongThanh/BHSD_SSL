#!/bin/bash
# ==============================================================================
# Script tự động submit Notebook Mean Teacher Dual GPU lên Kaggle bằng CLI
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"
KAGGLE_DIR="$PROJECT_ROOT/kaggle_mean_teacher"
KAGGLE_CLI="$HOME/.local/bin/kaggle"

echo "=========================================================================="
echo "=== SUBMIT MEAN TEACHER NOTEBOOK TO KAGGLE VIA CLI ==="
echo "=========================================================================="

# 1. Kiểm tra kaggle CLI
if [ ! -x "$KAGGLE_CLI" ] && ! command -v kaggle &> /dev/null; then
    echo "[!] Không tìm thấy lệnh kaggle. Đang cài đặt..."
    python3 -m pip install --user kaggle
fi

if [ -x "$KAGGLE_CLI" ]; then
    KCMD="$KAGGLE_CLI"
else
    KCMD="kaggle"
fi

# 2. Đồng bộ file notebook mới nhất từ scripts/ sang kaggle_mean_teacher/
echo "[*] Đồng bộ file notebook..."
cp "$PROJECT_ROOT/scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb" "$KAGGLE_DIR/kaggle_bhsd_mean_teacher_dual_gpu.ipynb"

# 3. Kiểm tra file metadata
if [ ! -f "$KAGGLE_DIR/kernel-metadata.json" ]; then
    echo "[X] Lỗi: Không tìm thấy $KAGGLE_DIR/kernel-metadata.json"
    exit 1
fi

echo "[*] Thông tin Kernel Metadata:"
cat "$KAGGLE_DIR/kernel-metadata.json"
echo ""

# 4. Thực hiện Push
echo "[*] Đang đẩy notebook lên Kaggle..."
"$KCMD" kernels push -p "$KAGGLE_DIR"

echo ""
echo "[✓] Đã submit thành công lên Kaggle!"
echo "[*] Kiểm tra trạng thái chạy bằng lệnh:"
echo "    $KCMD kernels status thnhnguynvncng/kaggle-bhsd-mean-teacher-dual-gpu"
echo "[*] Hoặc theo dõi trực tiếp tại:"
echo "    https://www.kaggle.com/code/thnhnguynvncng/kaggle-bhsd-mean-teacher-dual-gpu"
echo "=========================================================================="
