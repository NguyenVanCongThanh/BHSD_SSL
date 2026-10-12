"""
DEPRECATED:
Tập tin này thuộc về tập dữ liệu Lung Tumor (LT) trong mã nguồn nghiên cứu PICK gốc.
Nó đã được lưu trữ an toàn tại `legacy/LT_PICK_train.py`.

Vui lòng sử dụng điểm khởi chạy hợp nhất cho BHSD:
    python train.py --config configs/methods/pick.yaml
"""
import sys

if __name__ == "__main__":
    print("[DEPRECATION NOTICE] LT_PICK_train.py has been moved to legacy/LT_PICK_train.py.")
    print("For BHSD experiments, please use: python train.py --config configs/methods/pick.yaml")
    sys.exit(0)
