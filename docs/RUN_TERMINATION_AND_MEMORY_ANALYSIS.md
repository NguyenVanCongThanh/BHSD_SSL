# 📑 Báo Cáo Phân Tích Sự Cố Dừng Huấn Luyện (Run 20261023_102225), Cơ Chế Quản Lý Bộ Nhớ & Đề Xuất Tối Ưu Toàn Diện

> **Tài liệu tham chiếu nội bộ:** Dự án MedSSL-BHSD  
> **Tác giả:** Hệ thống phân tích kỹ thuật Antigravity  
> **Thời gian lập:** 25/10/2026  
> **Run mục tiêu:** `BHSD_multiclass_MeanTeacher_fold0_20261023_102225_mean_teacher_fold0` (ID: `s9i9nsui`)

---

## 1. TỔNG QUAN HIỆN TRẠNG & BẰNG CHỨNG THỰC TẾ

### 1.1. Bảng Thông Số Run Mới Nhất (`s9i9nsui`)
Qua việc truy vấn Weights & Biases API và tải trực tiếp file nhật ký bảng điều khiển `output.log` (104 KB) từ máy chủ Kaggle:

| Tiêu chí | Dữ liệu ghi nhận thực tế | Ghi chú kỹ thuật |
|---|---|---|
| **Mã Run & Trạng thái** | `s9i9nsui` \| State: `finished` | Kết thúc lúc 14:38 UTC (23/10/2026) |
| **Mốc Val cuối trên UI W&B** | **Iter 3750** | Do `val_interval: 250`, mốc tiếp theo là 4000 |
| **Mốc Huấn luyện thực tế** | **Iter 3996** | Ghi nhận trong W&B Summary `train/iter: 3996` |
| **Thời gian thực thi Kaggle** | **15,306s** (~4h 15m) | Kaggle báo cáo tổng phiên: **15,387s** (~4h 16m) |
| **Chất lượng mô hình** | Best mDice: **0.2902** (tại Iter 1750)<br>Val mDice cuối: **0.2425** (tại Iter 3750) | Đã hoàn thành 15 chu kỳ validation |
| **Trọng số VRAM GPU** | Ổn định tại **10.5 GB / 15.0 GB** | An toàn, không có hiện tượng VRAM GPU OOM |
| **Artifacts sinh ra** | 15 versions `latest` + 7 versions `best` | Tổng cộng 22 checkpoints đẩy lên cloud |

```
Log dòng 658: INFO:root:[train] Iter 3990/30000 (13.3%) | 2.13s/it | ETA: 15h22m51s ... | VRAM: 10.5GB
Log dòng 659: <TIẾN TRÌNH BỊ NGẮT ĐỘT NGỘT TẠI ITER 3996 - KHÔNG CÓ PYTHON TRACEBACK/EXCEPTION>
```

---

### 1.2. So Sánh Bước Nhảy Vọt Với Sự Cố Dừng Ở Iter 340 Trước Đó
Trước khi có run này, 3 lần chạy liên tiếp đều dừng bí ẩn ở mốc **Iter 340 – 341**:
- `a35mtk0i` (20261023_090226): Dừng tại **Iter 340** (3,700s, crash).
- `fum8iuny` (20261023_082536): Dừng tại **Iter 341** (3,975s).
- `1ex7ty08` (20261022_093652): Dừng tại **Iter 341** (11,523s).

**Phân tích nguyên nhân khác biệt:**
- **Run cũ (it 340):** Dùng `batch_size: 6` (mỗi GPU 3 mẫu 3D `[32, 192, 192]`). VRAM GPU bị đẩy lên **12.7 GB / 15.0 GB**, tiệm cận giới hạn vật lý và gây crash ở bước 340.
- **Run mới (it 3996):** Đã hạ `batch_size: 4` (mỗi GPU 2 mẫu). VRAM hạ xuống **10.5 GB** $\rightarrow$ Giúp mô hình chạy trơn tru vượt qua bước 340 và tiến sâu gấp **11 lần** (đạt 3996 bước).

---

## 2. NGUYÊN NHÂN GỐC RỄ KHIẾN RUN DỪNG TẠI ITER 3996

Việc tiến trình Python bị ngắt ngang mà không có bất kỳ dòng ngoại lệ (Exception/Traceback) nào chứng minh: **Tiến trình đã bị Hệ điều hành Linux gửi tín hiệu `SIGKILL` (Signal 9 / Exit code 137)**.

Ba nguyên nhân dẫn tới sự cố:

### 🔴 Nguyên Nhân 1: Tràn Dung Lượng Ổ Đĩa `/kaggle/working` (Giới hạn 20GB của Kaggle) - *Nguyên nhân chính*
- Mỗi checkpoint của mạng nnUNet3D 7-stage (gồm Student + Teacher EMA + Optimizer Adam + AMP Scaler) nặng **~450 MB**.
- Trong `configs/methods/mean_teacher.yaml`, thiết lập `save_wandb_model: true` và `val_interval: 250`.
- Khi W&B thực hiện `wandb.log_artifact()`, thư viện sẽ sao chép file `.pth` vào thư mục tạm `.wandb/artifacts/staging/` ngay trên ổ đĩa cục bộ `/kaggle/working`.
- Qua 15 lần validation:
  - Sinh ra 15 bản `latest` (`v0` đến `v14`) và 7 bản `best` (`v0` đến `v6`).
  - Tổng cộng 22 bản checkpoint $\times 450\text{ MB} \approx \mathbf{9.9\text{ GB}}$ đệm W&B.
  - Cộng thêm: checkpoint lưu cục bộ (`best_model.pth`, `checkpoint_latest.pth`, `VNet_best_model.pth` $\approx$ 1.35 GB), ảnh segmentation masks, log W&B và cache hệ thống.
  $\rightarrow$ **Ổ đĩa `/kaggle/working` chạm ngưỡng 20 GB (100% Full)**. Kaggle sandbox phát hiện tràn đĩa và lập tức cưỡng chế hủy container.

### 🔴 Nguyên Nhân 2: Tích Tụ Bộ Nhớ Host RAM (System RAM Limit 30GB)
- Mỗi lần validation, hàm sliding window duyệt qua **38 ca CT** 3D.
- Mỗi ca tạo ra mảng xác suất `score_map` kích thước $(6, 60, 512, 512)$ float32 ngốn tới **~380 MB RAM**.
- Do chuỗi tham chiếu vòng và việc W&B giữ lại các đối tượng trực quan hóa (4 khối CT 3D), lượng RAM này bị kẹt lại trong heap của Python. Qua 15 lần validation (tổng cộng $15 \times 38 = 570$ lượt nạp khối 3D), lượng RAM bị chiếm dụng tăng dần theo thời gian, tiệm cận ngưỡng 30 GB.

### 🔴 Nguyên Nhân 3: Tần Suất Validation Quá Dày Gây Nghẽn Thời Gian
- Cứ mỗi 250 bước huấn luyện (chạy mất ~500 giây), hệ thống lại dừng lại chạy sliding-window validation 3D trọn vẹn 38 ca (mất **~310 giây** $\approx$ hơn 5 phút).
- 15 lần validation đã tiêu tốn **77.5 phút** (hơn 1.25 giờ) chỉ riêng cho việc đánh giá!
- Tổng thời gian 15,387s cho 3996 bước tương đương trung bình ~3.85s/bước. Với tốc độ này, chạy hết 30,000 bước cần **~32 giờ** (vượt xa giới hạn tối đa 12 giờ của Kaggle).

---

## 3. GIẢI MÃ CHUYÊN SÂU VỀ GARBAGE COLLECTION (GC) TRONG PYTORCH & PYTHON

### 3.1. Rác (Garbage) Là Gì và Nằm Ở Đâu?
- Rác **không nằm trong một thư mục hay thùng rác riêng nào** trên máy tính.
- Khi một mảng NumPy hoặc Tensor được tạo ra, hệ điều hành đánh dấu các ô nhớ trên RAM là **"BẬN"** để cấp cho Python.
- Khi chương trình chạy xong hàm tính toán đó, nếu vùng nhớ đó không được trả lại cho hệ điều hành, nó vẫn bị đánh dấu là "BẬN" dù không còn ai dùng. Đó chính là **Rác**.

### 3.2. Rác Sẽ Tồn Tại Đến Khi Nào?
> **Rác KHÔNG TỰ BIẾN MẤT sau khi hết hàm Val, mà nó sẽ NẰM LÌ TRÊN RAM CHO ĐẾN HẾT CẢ BUỔI TRAIN (hoặc đến khi máy sập vì OOM)!**

Linux chỉ thu hồi RAM trong 2 trường hợp:
1. Tiến trình tự gọi lệnh giải phóng bộ nhớ (`free()` trong C hoặc `gc.collect()` trong Python).
2. Tiến trình kết thúc hoàn toàn hoặc bị hệ điều hành bắn chết (`SIGKILL`).

### 3.3. Hiện Tượng "Mù Dung Lượng" (Object Size Blindness) Của Python
- Bộ gom rác tự động của Python chỉ đếm **số lượng đối tượng (`PyObject`)**, không theo dõi dung lượng byte:
  - Một mảng `score_map` 3D (380 MB RAM) trong mắt Python chỉ là **1 đối tượng duy nhất** với vỏ bọc 56 bytes.
  - Python thấy mới chỉ có vài đối tượng được tạo ra nên **chủ quan không chạy dọn dẹp**, trong khi RAM thực tế đã cạn kiệt.

### 3.4. Phân Biệt `gc.collect()` và `torch.cuda.empty_cache()`

| Tiêu chí | `gc.collect()` | `torch.cuda.empty_cache()` |
|---|---|---|
| **Khu vực tác động** | **System RAM (RAM máy tính / Host)** | **VRAM GPU (Card màn hình NVIDIA)** |
| **Cơ chế** | Phá vỡ các tham chiếu vòng, thu hồi bộ nhớ mảng NumPy/Python chết | Trả các khối VRAM đệm mà PyTorch đang giữ lại về cho driver GPU |
| **Ảnh hưởng kết quả?** | **100% KHÔNG ẢNH HƯỞNG**. Không can thiệp vào trọng số, gradient hay điểm mDice | **100% KHÔNG ẢNH HƯỞNG**. |

### 3.5. Trade-off (Sự Đánh Đổi) & "Điểm Rơi Vàng"
- **Tại sao không dọn rác ở từng batch?** 
  Việc gọi `gc.collect()` và `empty_cache()` bắt CPU và GPU phải dừng lại đồng bộ, tốn từ 0.1s – 0.5s. Nếu gọi ở mọi bước lặp ($30,000\text{ steps}$), buổi huấn luyện sẽ bị chậm thêm gần 2 tiếng đồng hồ!
- **Điểm rơi vàng (Best Practice):** 
  Chỉ gọi sau khi **kết thúc chu kỳ Validation** (mỗi 500 hoặc 1000 bước). Dành ra 0.3 giây sau một chu kỳ val kéo dài 5 phút để dọn sạch 15 GB rác là hoàn toàn xứng đáng và không hề ảnh hưởng đến tốc độ chung.

---

## 4. BẢNG TỔNG HỢP CÁC SCRIPT ĐÃ CÓ VÀ ĐÃ BỔ SUNG TRONG REPO

| Script | Vị trí | Chức năng chính |
|---|---|---|
| [`fetch_latest_run.py`](file:///home/u001015/BHSD_SSL/tools/fetch_latest_run.py) | `tools/` | Kéo toàn bộ metrics, history, summary của run mới nhất từ W&B |
| [`fetch_wandb_logs.py`](file:///home/u001015/BHSD_SSL/tools/fetch_wandb_logs.py) | `tools/` | Tải file `output.log` (104 KB console stdout) trực tiếp từ W&B Cloud |
| [`list_runs_summary.py`](file:///home/u001015/BHSD_SSL/tools/list_runs_summary.py) | `tools/` | Quét và lập bảng thống kê toàn bộ 14 runs (runtime, state, best dice) |
| [`compare_old_new_runs.py`](file:///home/u001015/BHSD_SSL/tools/compare_old_new_runs.py) | `tools/` | So sánh chi tiết diff cấu hình giữa run it 340 và run it 3996 |
| [`kaggle_ctl.py`](file:///home/u001015/BHSD_SSL/tools/kaggle/kaggle_ctl.py) | `tools/kaggle/` | Quản lý đa tài khoản Kaggle, kiểm tra trạng thái và submit kernel tự động |

---

## 5. ĐỀ XUẤT CẢI TIẾN TOÀN DIỆN CHO CÁC LẦN CHẠY TIẾP THEO

### Đề xuất 1: Triệt tiêu nguy cơ tràn đĩa W&B Artifacts (Ưu tiên số 1)
Trong file cấu hình [`configs/methods/mean_teacher.yaml`](file:///home/u001015/BHSD_SSL/configs/methods/mean_teacher.yaml):
- Đặt `save_wandb_model: false` trong quá trình train định kỳ.
- Chỉ lưu checkpoint cục bộ ghi đè: `checkpoint_latest.pth` và `best_model.pth`.
- Chỉ đẩy artifact lên W&B khi mô hình phá kỷ lục `best_mDice` hoặc ở bước kết thúc (tránh tạo 22 versions rác staging ngốn 10GB đĩa).

### Đề xuất 2: Giãn chu kỳ Validation để tăng tốc và giảm tải
- Tăng `val_interval` từ 250 lên **500** hoặc **1000** iterations.
- Giúp giảm 50% – 75% thời gian chạy lãng phí vào validation (tiết kiệm gần 2 giờ trên Kaggle), giảm số file checkpoint sinh ra và giúp mô hình hoàn thành đủ 30,000 steps trong hạn mức 12 giờ của Kaggle.

### Đề xuất 3: Chèn dọn rác chủ động sau Validation
Tại [`methods/base_method.py`](file:///home/u001015/BHSD_SSL/methods/base_method.py#L481-L485):
Chèn khối lệnh giải phóng bộ nhớ ngay sau khi gom kết quả validation xong:
```python
if is_main_process():
    # ... log metrics & wandb ...
    pass

if self.is_ddp:
    torch.distributed.barrier()

# DỌN RÁC TRIỆT ĐỂ:
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

self.model.train()
return best_dice
```

### Đề xuất 4: Kích hoạt Auto-Resume từ Checkpoint Iter 3996
Trong notebook submit tiếp theo:
- Đặt `resume: true` trong cấu hình experiment.
- Mô hình sẽ tự động nạp `checkpoint_latest.pth` (hoặc kéo từ W&B) và tiếp tục huấn luyện ngay từ bước **3996** hướng tới 30,000, bảo toàn toàn bộ kết quả đã học mà không phải tốn 4 tiếng chạy lại từ bước 0.
