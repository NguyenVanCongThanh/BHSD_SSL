# 📘 Báo Cáo Kỹ Thuật: Khắc Phục Hiện Tượng Treo & Không Xuất Log Khi Save Run / CLI Push Trên Kaggle Dual GPU (2x T4)

*Tài liệu ghi chép chi tiết nguyên nhân gốc rễ, các giải pháp phòng ngừa (Anti-Deadlock / Realtime Streaming) và các thay đổi đã triển khai trong MedSSL-BHSD.*

---

## 1. Hiện Tượng (Symptoms)
- Khi bấm **Save & Run All (Commit)** hoặc chạy lệnh submit qua CLI (`kaggle kernels push`), notebook khởi chạy được vài phút (ví dụ: `Running for 161.1s` hay 3-5 phút) nhưng mục **Logs** hoàn toàn trống trơn hoặc chỉ có `[]`.
- Biểu đồ **Weights & Biases (W&B)** hoàn toàn không có tín hiệu (chưa nạp tới bước 5).
- Tình trạng này diễn ra lặp lại nhiều lần liên tiếp, nhưng đôi khi Cancel và submit lại ngẫu nhiên thì có lúc lại chạy được.

---

## 2. Nguyên Nhân Gốc Rễ (Root Causes)

### A. Deadlock do Khởi Tạo PyTorch DDP / NCCL trên Container Ảo Hóa
1. **Thiếu timeout cho NCCL Process Group**:
   - Trong `core/distributed.py`, hàm `dist.init_process_group` mặc định không cấu hình `timeout` (PyTorch mặc định chờ 30 phút hoặc vô hạn).
   - Nếu 2 process GPU (Rank 0 và Rank 1) bị chậm trễ kết nối, cả 2 sẽ đứng đợi nhau vô hạn tại `dist.barrier()`.
2. **Xung đột cổng `MASTER_PORT=29500`**:
   - Khi chạy qua `torchrun`, port mặc định `29500` dễ bị chiếm dụng hoặc chưa kịp giải phóng sau lần cancel trước đó trên cùng container máy chủ.
3. **Lỗi PCIe P2P Direct Memory trên Kaggle T4**:
   - Hai card GPU T4 của Kaggle nằm trên bus PCIe ảo hóa. NCCL cố gắng bắt tay bằng giao thức Peer-to-Peer (P2P direct memory) có thể bị driver ảo của Kaggle chặn lại gây nghẽn (hang) im lặng.

### B. Buffering của Stdout / Stderr & Lệnh Chạy Ngầm
1. **Pip install với cờ `-q` (Quiet)**:
   - `%pip install -q ...` triệt tiêu toàn bộ tiến trình tải và compile các gói C nặng (`SimpleITK`, `medpy`). Khi mạng Kaggle phân giải package chậm, giao diện Logs của Kaggle không nhận được dòng text nào nên nhìn như bị đơ.
2. **Buffer dòng của Subprocess Python**:
   - Lệnh `print()` và `subprocess.Popen` nếu thiếu `flush=True` sẽ bị Python giữ lại trong bộ đệm nội bộ (I/O Buffer) cho đến khi đầy buffer mới đẩy ra màn hình, khiến log bị trễ nghiêm trọng.
3. **Git Clone bị chặn prompt ẩn (`GIT_TERMINAL_PROMPT`)**:
   - Trong môi trường chạy nền (non-interactive shell), nếu có bất kỳ cảnh báo hoặc prompt xác thực ngầm nào từ Git, tiến trình sẽ rơi vào trạng thái chờ nhập vô hạn.

---

## 3. Các Giải Pháp & Thay Đổi Đã Triển Khai (Implemented Solutions)

### 3.1. Cập nhật mã nguồn hệ thống: [`core/distributed.py`](file:///home/u001015/BHSD_SSL/core/distributed.py)
* **Thêm NCCL Timeout 120s**:
  ```python
  from datetime import timedelta
  dist.init_process_group(
      backend=dist_backend,
      init_method="env://",
      world_size=world_size,
      rank=rank,
      timeout=timedelta(seconds=120)  # Tự động ném lỗi nếu quá 2 phút không bắt tay được
  )
  ```
* **Thêm log chi tiết từng chặng của từng Rank (`flush=True`)**:
  * In rõ `[DDP-Rank X] Đang khởi tạo...`, `Đang đồng bộ barrier...`, `Khởi tạo DDP thành công!`. Giúp debug chính xác vị trí bị nghẽn.

### 3.2. Cập nhật Notebook Kaggle & Generator Tool:
*Files liên quan: [`kaggle_mean_teacher/kaggle_bhsd_mean_teacher_dual_gpu.ipynb`](file:///home/u001015/BHSD_SSL/kaggle_mean_teacher/kaggle_bhsd_mean_teacher_dual_gpu.ipynb), [`scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb`](file:///home/u001015/BHSD_SSL/scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb), [`tools/generate_mean_teacher_notebook.py`](file:///home/u001015/BHSD_SSL/tools/generate_mean_teacher_notebook.py)*.

1. **Bước 1: Bật thanh tiến trình trực quan cho Pip**:
   ```bash
   !pip install --progress-bar on medpy tensorboardX h5py SimpleITK nibabel wandb pyyaml
   ```
2. **Bước 2: Chống nghẽn Git**:
   ```python
   env["GIT_TERMINAL_PROMPT"] = "0"
   subprocess.run(["git", "clone", ...], env=env, timeout=120)
   ```
3. **Bước 3: Định vị Dataset BHSD tức thì**:
   * Tối ưu hóa hàm `find_bhsd_root` để nhận diện ngay thư mục `BHSD_fold0` trong `thnhnguynvncng/bhsd-fold0` trong < 1 giây, bỏ qua việc quét đệ quy tar chậm chạp.
4. **Bước 6: Anti-Deadlock cho Torchrun & NCCL**:
   * **Cấp phát Port ngẫu nhiên**:
     ```python
     import socket
     def find_free_port():
         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
             s.bind(('', 0))
             return s.getsockname()[1]
     master_port = str(find_free_port())
     ```
   * **Tối ưu biến môi trường NCCL**:
     ```python
     env["NCCL_DEBUG"] = "WARN"
     env["NCCL_IB_DISABLE"] = "1"     # Tắt InfiniBand
     env["NCCL_P2P_DISABLE"] = "1"    # Tắt P2P Direct tránh treo PCIe ảo hóa
     env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
     ```
   * **Stream Output với `flush=True`**:
     ```python
     for line in process.stdout:
         print(line, end="", flush=True)
     ```

---

## 4. Checklist Kiểm Tra Nhanh Cho Các Notebook Mới Trong Tương Lai
Mỗi khi tạo hoặc sao chép một notebook chạy Dual GPU (2x T4 / 2x P100) trên Kaggle, luôn đảm bảo:
- [x] Không dùng `-q` trong pip install (`--progress-bar on`).
- [x] Cấu hình `GIT_TERMINAL_PROMPT=0` kèm timeout cho mọi lệnh gọi subprocess hệ thống.
- [x] Dùng `find_free_port()` cho `--master_port` thay vì cố định `29500`.
- [x] Bật `NCCL_P2P_DISABLE=1` và `NCCL_IB_DISABLE=1` khi chạy DDP trên môi trường máy ảo đám mây.
- [x] Cấu hình `timeout=timedelta(seconds=120)` trong `dist.init_process_group`.
- [x] Đảm bảo mọi lệnh `print()` quan trọng có tham số `flush=True`.
