# Submit Kaggle và quản lý tài khoản

Đối chiếu công cụ ngày 29/10/2026. Controller dựng bundle theo notebook và username, dùng profile tài khoản riêng. Trạng thái và tài nguyên được cấp thực tế cần kiểm tra trên phiên Kaggle.

## 1. Cấu Trúc Hệ Thống Mới

Notebook Mean Teacher chuẩn nằm ở [`scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb`](../../scripts/kaggle_bhsd_mean_teacher_dual_gpu.ipynb).
Metadata được tạo từ [`configs/kaggle/notebooks.json`](../../configs/kaggle/notebooks.json) trong bundle `kaggle_runs/<notebook>/<username>/`; `kaggle_mean_teacher/` không còn là nguồn notebook chuẩn; thư mục generated/legacy vẫn có thể tồn tại local, không suy việc xóa local từ trạng thái Git.
Để tạo và kiểm tra bundle cục bộ trước khi submit:

```bash
python tools/kaggle/kaggle_ctl.py build --notebook mean_teacher --username dry_run_user
```

Môi trường chạy controller cần `kaggle` CLI/API và `requests`, ngoài dependency của từng công cụ trong `requirements-tools.txt`.
Các file requirements liệt kê dependency và cận dưới tham khảo; chưa có lockfile hay xác nhận mọi tổ hợp phiên bản/CUDA đều tương thích.
Khi tái lập thực nghiệm, lưu phiên bản thực tế đã kiểm chứng của Python, PyTorch/torchvision, CUDA và các package liên quan.

```
BHSD_SSL/
├── configs/kaggle/
│   ├── accounts.json          # Registry quản lý danh sách tài khoản (ID, username, API Key)
│   ├── notebooks.json         # Danh mục các loại notebook (Mean Teacher, PICK, GPU/Internet settings)
│   └── secrets/               # Thư mục cache profile cô lập cho từng tài khoản (chmod 600, đã được ignore khỏi Git)
├── tools/kaggle/
│   ├── accounts_manager.py    # Quản lý đọc, lưu, tạo isolated profile directory cho từng user
│   ├── notebook_builder.py    # Tự động chuẩn bị notebook bundle và kernel-metadata.json theo từng user
│   ├── kaggle_executor.py     # Sandbox thực thi Kaggle CLI/API độc lập, an toàn
│   └── kaggle_ctl.py          # CLI điều khiển trung tâm (account, push, status)
├── kaggle_runs/               # Thư mục chứa các bundle đã sinh cho từng user trước khi đẩy lên Kaggle
└── scripts/
    ├── submit_mean_teacher_kaggle.sh  # Script nộp Mean Teacher (tương thích ngược hoàn toàn)
    └── kaggle_submit.sh               # Script nộp tổng quát cho mọi notebook
```

---

## 2. Danh Sách Tài Khoản Đang Quản Lý

Xem danh sách tài khoản hiện có:
```bash
python3 tools/kaggle/kaggle_ctl.py account list
# Hoặc:
bash scripts/submit_mean_teacher_kaggle.sh --list-accounts
```

Danh sách tài khoản thực tế thuộc cấu hình local, không được đóng băng trong tài liệu. Dùng lệnh trên để tra ID/slug/username trước khi submit. Các ID 1/2 và username trong ví dụ phía dưới chỉ minh họa cấu hình dự án trước đây; thay bằng registry của bạn.

---

## 3. Cách Submit Notebook Lên Kaggle

### Cách 1: Sử dụng Shell Script quen thuộc (`submit_mean_teacher_kaggle.sh`)

- **Nộp bằng Tài khoản 1 (Thành - Mặc định)**:
  ```bash
  bash scripts/submit_mean_teacher_kaggle.sh
  ```

- **Nộp bằng Tài khoản 2 (Khánh)**:
  ```bash
  bash scripts/submit_mean_teacher_kaggle.sh --account 2
  ```

- **Ghi đè Dataset nguồn (nếu cần)**:
  ```bash
  bash scripts/submit_mean_teacher_kaggle.sh --account 2 --dataset thnhnguynvncng/bhsd-fold0
  ```

- **Kiểm tra trạng thái đang chạy trên Kaggle**:
  ```bash
  bash scripts/submit_mean_teacher_kaggle.sh --account 2 --status
  ```

---

### Cách 2: Sử dụng CLI Điều Khiển Trung Tâm (`kaggle_ctl.py`)

- **Submit Mean Teacher Dual GPU**:
  ```bash
  # Submit cho Tài khoản 1
  python3 tools/kaggle/kaggle_ctl.py push --account 1 --notebook mean_teacher

  # Submit cho Tài khoản 2
  python3 tools/kaggle/kaggle_ctl.py push --account 2 --notebook mean_teacher
  ```

- **Submit PICK Dual GPU**:
  ```bash
  python3 tools/kaggle/kaggle_ctl.py push --account 1 --notebook pick
  python3 tools/kaggle/kaggle_ctl.py push --account 2 --notebook pick
  ```

- **Kiểm tra trạng thái Kernel**:
  ```bash
  python3 tools/kaggle/kaggle_ctl.py status --account 1 --notebook mean_teacher
  python3 tools/kaggle/kaggle_ctl.py status --account 2 --notebook mean_teacher
  ```

---

## 4. Hướng Dẫn Mở Rộng Sau Này

### 4.1. Thêm Tài Khoản Mới (Ví dụ Tài khoản 3)
Chỉ cần chạy lệnh:
```bash
python3 tools/kaggle/kaggle_ctl.py account add \
    --id 3 \
    --username <ten_kaggle_username> \
    --key KGAT_xxxxxxxxxxxxxxxxxxxxxxxx \
    --name "Tài khoản 3 - [Tên bạn]" \
    --slug user3
```
Sau đó bạn có thể submit ngay lập tức:
```bash
python3 tools/kaggle/kaggle_ctl.py push --account 3 --notebook mean_teacher
```

### 4.2. Thêm Loại Notebook Mới
Mở file `configs/kaggle/notebooks.json` và thêm cấu hình:
```json
{
  "notebooks": {
    "my_new_method": {
      "slug": "kaggle-bhsd-my-method",
      "title": "kaggle_bhsd_my_method",
      "code_file": "kaggle_bhsd_my_method.ipynb",
      "generator_script": "tools/generate_my_method.py",
      "enable_gpu": true,
      "enable_internet": true,
      "dataset_sources": ["thnhnguynvncng/bhsd-fold0"]
    }
  }
}
```
Lệnh submit sẽ tự động nhận diện:
```bash
python3 tools/kaggle/kaggle_ctl.py push --account 1 --notebook my_new_method
```

---

## 5. Nguyên Lý Tự Động Khởi Tạo Notebook Mới Trên Tài Khoản 2
1. Khi gọi `push --account 2`, `NotebookBuilder` sẽ tạo thư mục bundle cô lập tại:
   `kaggle_runs/mean_teacher/bokhnhl/`
2. Bundle bao gồm file notebook nguồn mới nhất và `kernel-metadata.json` với trường:
   `"id": "bokhnhl/kaggle-bhsd-mean-teacher-dual-gpu"`
3. Kaggle API nhận request với đúng username của phiên xác thực. Nếu notebook chưa có trên tài khoản của Khánh, Kaggle server sẽ **tự động khởi tạo mới (Create)** và yêu cầu cấu hình GPU theo metadata. Việc tạo/chạy, loại GPU và quota thực tế cần xác minh từ phản hồi nền tảng; không bảo đảm luôn nhận 2x T4.
4. Đường link trực tiếp sau khi submit:
   - Tài khoản 1: `https://www.kaggle.com/code/thnhnguynvncng/kaggle-bhsd-mean-teacher-dual-gpu`
   - Tài khoản 2: `https://www.kaggle.com/code/bokhnhl/kaggle-bhsd-mean-teacher-dual-gpu`

## Tài liệu liên quan

Xem [troubleshooting](kaggle-troubleshooting.md), [batch DDP](training.md) và [khôi phục checkpoint](checkpoint-resume.md).
