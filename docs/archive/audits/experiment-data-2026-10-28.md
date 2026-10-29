> **Hồ sơ lịch sử:** giữ nguyên phát hiện, số liệu và kết quả kiểm tra tại thời điểm báo cáo; không dùng các mô tả “hiện tại” dưới đây làm trạng thái checkout mới. Xem [chỉ mục tài liệu](../../README.md) để tìm hướng dẫn hiện hành.

# Báo cáo rà soát và Kiến trúc chuẩn hóa thư mục Experiment Data

*Tài liệu thẩm định, dọn dẹp và chuẩn hóa kiến trúc lưu trữ dữ liệu thí nghiệm (`experiment_data/`)*  
*Thời gian thực hiện: 28/10/2026*

---

## 1. Tóm tắt kết quả rà soát hiện trạng (Executive Summary)

Thư mục `experiment_data/` hiện đang chiếm khoảng **39 MB** dung lượng lưu trữ cục bộ, chứa dữ liệu thu thập từ 2 nền tảng: **Weights & Biases (W&B)** và **Kaggle Kernel Sessions**.

Qua rà soát chi tiết, cấu trúc hiện tại xuất hiện cảm giác "rối" và khó quản lý do các nguyên nhân chính:
1. **Phân mảnh vị trí lưu trữ**: Tồn tại song song cả `wandb/<run_id>/` (chứa snapshot JSON thô lồng nhau nhiều cấp) và `wandb_exports/<run_id>/` (chứa bảng số liệu CSV/JSON phẳng).
2. **Hiện tượng Snapshot Spam**: Một số script tự động (như `fetch_wandb.py`) mỗi lần chạy đều tạo thêm snapshot mới cho các run đích, sinh ra các thư mục con chỉ cách nhau vài giây với nội dung trùng lặp.
3. **Độ dài đường dẫn và chuỗi định danh ngẫu nhiên**: Mỗi snapshot mang tên dạng `YYYYMMDDTHHMMSS_ffffffZ_<uuid8>` (ví dụ: `20261028T050921_123714Z_840978e3`), khiến người dùng không thể gõ tay hoặc truy cập nhanh trong terminal.
4. **Mã nguồn và tàn tích migration cũ nằm lẫn trong dữ liệu**: Có 5 file mã nguồn Python được sao chép vào `kaggle/.../source/` từ đợt kiểm toán sự cố ngày 26/10 và thư mục `imports/migration/` ghi lại bảng băm chuyển dịch file.

---

## 2. Bảng kiểm kê chi tiết toàn bộ thư mục `experiment_data/`

| Đường dẫn thư mục con | Số file | Dung lượng | Loại dữ liệu | Đánh giá & Khuyến nghị |
| :--- | :---: | :---: | :--- | :--- |
| **`wandb/9tvq8qld/`** | 7 files | **21.5 MB** | **Snapshot toàn diện run mới nhất (16.250 iters)**: `history.json` (6.2MB), `events.json` (15MB), `config.json`, `summary.json`, `artifacts.json`, `wandb-metadata.json`. | **Tài sản quan trọng nhất**. Đã tạo symlink `latest` để truy cập tức thì. Đã xóa bỏ snapshot trùng lặp 63KB sinh ra sau đó 40s. |
| **`wandb_exports/9tvq8qld/`** | 2 files | **6.6 MB** | **Dữ liệu bảng phẳng**: `metrics_history.csv` (474 KB - 1.690 dòng, 87 cột) và `metrics_history.json` (6.1 MB). | **Rất cần thiết** cho Pandas, Excel và vẽ đồ thị khoa học. |
| **`wandb/s9i9nsui/`** | 8 files | **4.9 MB** | Snapshot hoàn chỉnh run Mean Teacher 3.996 iters (ngày 23/10). | **Lưu trữ làm mốc so sánh lịch sử**. |
| **`wandb/1ex7ty08/`** | 4 files | **3.6 MB** | System events, log, metadata của run 341 iters (ngày 22/10). | Giữ lại (dùng cho phân tích OOM/Crash phần cứng trong doc sự cố). |
| **`wandb/fum8iuny/`** | 4 files | **1.2 MB** | System events, log, metadata của run 341 iters (ngày 23/10). | Giữ lại (phục vụ đối chiếu sự cố). |
| **`wandb/a35mtk0i/`** | 2 files | **860 KB** | System events của run 340 iters (ngày 23/10). | Giữ lại (phục vụ đối chiếu sự cố). |
| **`wandb/5i9jspgj/`** | 4 files | **45 KB** | Baseline chuẩn 12.000 iters (ngày 16/10). | **Cần thiết** làm mốc so sánh (Baseline). |
| **`wandb/lgvtexrb/`** | 4 files | **14 KB** | Run liền trước (ngày 17/10). | **Cần thiết** làm mốc so sánh (Previous). |
| **`wandb/kpu7le0o/`** | 3 files | **13 KB** | Snapshot cũ của run ngày 18/10. | Lưu trữ tham khảo. |
| **`wandb/_collections/`** | 5 files | **204 KB** | Gồm 2 snapshot: `20261026...` (cũ) và `20261028...` (mới nhất chứa `9tvq8qld`). | Đã tạo symlink `latest` trỏ thẳng vào snapshot mới nhất. |
| **`kaggle/thnhnguynvncng/`** | 9 files | **416 KB** | Output log, files metadata và 5 file source code đối chiếu. | Giữ lại log; 5 file source code là tàn tích audit có thể dọn dẹp hoặc đóng băng. |
| **`kaggle/bokhnhl/`** | 3 files | **48 KB** | Output log và files metadata phiên chạy trên tài khoản phụ. | Giữ lại log tham khảo. |
| **`imports/migration/`** | 2 files | **10.5 KB** | Bảng băm SHA-256 các file đã di chuyển trong đợt dọn dẹp ngày 26/10. | Giữ lại làm nhật ký kiểm toán (audit trail). |

---

## 3. Các hành động xử lý và tối ưu hóa đã thực hiện ngay

1. **Xóa snapshot rác trùng lặp**:
   - Đã loại bỏ thư mục `wandb/9tvq8qld/20261028T051006_176623Z_97791eef/` (chỉ chứa 4 file nhỏ trùng lặp sinh ra ngoài ý muốn khi chạy `fetch_wandb.py`), giữ lại duy nhất bản snapshot gốc 21.5 MB đầy đủ.
2. **Triển khai cơ chế Symlink `latest` tự động**:
   - Nâng cấp module trung tâm [`tools/crawl_storage.py`](../../../tools/crawl_storage.py): Mỗi khi tạo snapshot mới, hệ thống tự động cập nhật liên kết biểu tượng `latest` trong thư mục cha.
   - Giờ đây bạn có thể mở nhanh bất kỳ file nào mà không cần nhớ timestamp:
     - `cat experiment_data/wandb/9tvq8qld/latest/summary.json`
     - `cat experiment_data/wandb/9tvq8qld/latest/config.json`
     - `python3 tools/compare_runs.py --input experiment_data/wandb/_collections/latest/wandb_results.json`
3. **Thống nhất luồng xuất dữ liệu bảng**:
   - Dữ liệu dạng bảng phục vụ phân tích khoa học / vẽ đồ thị luôn được định tuyến về `experiment_data/wandb_exports/<run_id>/metrics_history.csv` để giữ đường dẫn phẳng, dễ truy cập bằng Pandas.

---

## 4. Kiến trúc chuẩn hóa đề xuất (Chuẩn nhất & Dễ mở rộng nhất)

Để hệ thống luôn ngăn nắp khi bạn chạy thêm hàng chục thí nghiệm nữa, cấu trúc `experiment_data/` được chuẩn hóa thành 3 phân tầng chức năng rõ rệt:

```text
experiment_data/
│
├── wandb/                               # [TẦNG 1: DỮ LIỆU W&B NGUYÊN BẢN]
│   ├── <run_id>/                        # Mỗi run là 1 thư mục riêng theo ID W&B
│   │   ├── latest -> <snapshot_newest>  # Symlink trỏ nhanh đến snapshot mới nhất
│   │   └── <timestamp_snapshot>/        # Snapshot lưu trữ bất biến (immutable)
│   │       ├── config.json              # Toàn bộ hyperparameters
│   │       ├── summary.json             # Chỉ số best và final
│   │       ├── history.json             # Toàn bộ metric logs qua các iteration
│   │       ├── events.json              # Giám sát GPU/VRAM/CPU theo thời gian
│   │       ├── output.log               # Console log từ W&B
│   │       ├── wandb-metadata.json      # Môi trường chạy, GPU model, commit SHA
│   │       └── artifacts.json           # Danh sách model checkpoints
│   │
│   └── _collections/                    # [TẦNG 2: BỘ DỮ LIỆU SO SÁNH OFFLINE]
│       ├── latest -> <collection_newest># Symlink bộ sưu tập mới nhất
│       └── <timestamp>/                 
│           └── wandb_results.json       # Tập hợp Recent + Prev + Baseline
│
├── wandb_exports/                       # [TẦNG 3: BẢNG SỐ LIỆU PHẲNG CHO PHÂN TÍCH]
│   └── <run_id>/                        
│       ├── metrics_history.csv          # Bảng phẳng (87 cột) cho Pandas / Excel
│       └── metrics_history.json         # Bản JSON phẳng đã lọc bỏ media binary
│
├── kaggle/                              # [TẦNG 4: LOG VÀ PHIÊN CHẠY KAGGLE]
│   └── <username>/
│       └── <kernel_slug>/
│           ├── latest -> <snapshot_newest>
│           └── <timestamp>/
│               ├── output.log.json      # Raw log từ Kaggle API
│               ├── status.json          # Trạng thái kết thúc của Kernel
│               └── files.json           # Danh sách file sinh ra trong /kaggle/working
│
└── imports/                             # [TẦNG 5: NHẬT KÝ KIỂM TOÁN LỊCH SỬ]
    └── migration/                       # Ghi nhận lịch sử di dời file (audit trail)
```

---

## 5. Quy chuẩn thao tác dành cho người dùng (Quick Reference)

| Nhu cầu của bạn | Lệnh thực hiện | File kết quả nhận được |
| :--- | :--- | :--- |
| **Kéo trọn bộ 1 run mới từ W&B** | `python3 tools/crawl_run.py ENTITY/PROJECT/<run_id>` | `experiment_data/wandb/<run_id>/latest/` |
| **Xuất bảng CSV để vẽ biểu đồ / Excel** | `python3 tools/export_wandb_history.py --run_id <run_id>` | `experiment_data/wandb_exports/<run_id>/metrics_history.csv` |
| **Cập nhật dữ liệu so sánh 3 run** | `python3 tools/fetch_wandb.py` | `experiment_data/wandb/_collections/latest/wandb_results.json` |
| **Xem bảng so sánh offline trên Terminal** | `python3 tools/compare_runs.py` | Bảng in trực tiếp ra terminal |
| **Kéo riêng file log phiên chạy Kaggle** | `bash scripts/kaggle_submit.sh --log -a 1 -n mean_teacher` | `logs/<kernel_slug>.log` |

---

## 6. Chính sách dọn dẹp định kỳ (Retention & Pruning Policy)

Khi ổ đĩa đầy hoặc qua nhiều tuần thí nghiệm:
1. **Các file an toàn để xóa bớt**:
   - `events.json` trong các snapshot cũ (chiếm tới 15 MB/run, chỉ cần khi điều tra lỗi quá nhiệt/OOM).
   - Các collection cũ trong `wandb/_collections/` (chỉ cần giữ snapshot mà `latest` đang trỏ tới).
   - `metrics_history.json` trong `wandb_exports/<run_id>/` (nếu đã có file `.csv`, file `.json` này có thể tạo lại bất cứ lúc nào).
2. **Các file TUYỆT ĐỐI NÊN GIỮ**:
   - `summary.json`, `config.json`, `manifest.json` (rất nhẹ, vài KB, là chứng cứ cấu hình).
   - `metrics_history.csv` (chỉ vài trăm KB nhưng chứa trọn vẹn kết quả chạy).
   - Snapshot của run Baseline (`5i9jspgj`) và run có kết quả tốt nhất (`9tvq8qld`).

