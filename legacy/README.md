# Legacy Code Archive

Thư mục này lưu trữ toàn bộ mã nguồn cũ kế thừa từ nghiên cứu PICK gốc trên các tập dữ liệu không thuộc BHSD:
- **Left Atrium (LA)**: `LA_PICK_train.py`, `dataloaders/la_heart_processing.py`
- **Lung Tumor (LT)**: `LT_PICK_train.py`, `dataloaders/dataset_lu.py`, `dataloaders/lung_tumor_processing.py`, `dataloaders/lung_split.py`
- **Pancreas (PA)**: `PA_PICK_train.py`, `dataloaders/dataset_pa.py`
- **ACDC**: `dataloaders/acdc_data_processing.py`
- **Gốc khác**: `networks/git_VNet.py` (bản sao gốc của VNet), `test_util.py`, `dataloaders/io_.py`

Các file này được giữ lại nhằm mục đích đối chiếu và tham khảo lịch sử, không tham gia vào luồng huấn luyện chính của framework MedSSL-BHSD.
Mọi huấn luyện và đánh giá trên tập dữ liệu BHSD được chuyển sang kiến trúc mô-đun mới:
- Điểm chạy huấn luyện: `train.py`
- Điểm đánh giá: `evaluate.py`
- Cấu hình: `configs/`
- Thuật toán: `methods/`
- Dữ liệu: `data/`
