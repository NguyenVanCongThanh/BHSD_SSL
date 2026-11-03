# Xử lý Lệch Task (Multiclass vs Binary) Khi Resume Trên Kaggle

> **Tài liệu kỹ thuật & Hướng dẫn khắc phục sự cố**  
> *Ngày lập: Tháng 11/2026*  
> *Liên quan: `tools/kaggle/kaggle_ctl.py`, `tools/kaggle/notebook_builder.py`, `tools/kaggle/resume.py`, `methods/base_method.py`, `utils/checkpoint.py`*

---

## 1. Bối cảnh & Tình huống Sự cố

Trong quá trình quản lý huấn luyện bán giám sát (SSL - Mean Teacher) qua công cụ CLI `kaggle_ctl.py`, người dùng có thể thực hiện lệnh nối tiếp (resume) một thí nghiệm trước đó:

```bash
python3 tools/kaggle/kaggle_ctl.py push \
  --account 1 \
  --notebook mean_teacher \
  --task binary \
  --resume-run <multiclass_run_id> \
  --resume-checkpoint best \
  --resume-mode same-run
```

**Tình huống đặt ra:**  
Run nguồn `<multiclass_run_id>` trên Weights & Biases (W&B) vốn là bài toán **Multiclass** (6 phân lớp: Background + 5 loại xuất huyết não), nhưng câu lệnh resume ở CLI lại chỉ định `--task binary` (2 phân lớp: Background + Xuất huyết chung).

**Câu hỏi cốt lõi:**  
Hệ thống hiện tại có cơ chế chặn ngay từ đầu hay không, hay vẫn cho phép submit và sẽ bị crash trong lúc chạy?

---

## 2. Phân tích Hiện trạng Mã nguồn (Root Cause Analysis)

### 2.1. Ở tầng dòng lệnh CLI (`kaggle_ctl.py` / `NotebookBuilder`)
* **Trạng thái:** **CHƯA CÓ cơ chế chặn (validation)**.
* **Chi tiết:** Trong [`tools/kaggle/notebook_builder.py`](file:///home/u001015/BHSD_SSL/tools/kaggle/notebook_builder.py#L58-L65), CLI chỉ thực hiện kiểm tra cú pháp cơ bản của tham số:
  * Kiểm tra cờ `--resume-checkpoint` / `--resume-mode` có đi kèm `--resume-run` hay không.
  * Kiểm tra tính hợp lệ của checkpoint (`latest`, `best`) và mode (`same-run`, `new-run`).
  * CLI **hoàn toàn không truy vấn W&B API** để kiểm tra tính tương thích giữa cờ `--task` và cấu hình thực tế của run nguồn.
* **Hệ quả:** Lệnh `push` vẫn báo thành công `[✓ THÀNH CÔNG] Notebook đã được đẩy lên Kaggle`, bundle notebook vẫn được khởi tạo và gửi lên hàng đợi Kaggle.

---

### 2.2. Ở tầng thực thi trên Kaggle (`prepare_resume` trong `tools/kaggle/resume.py`)
* **Trạng thái:** **KHÔNG BỊ CRASH, nhưng xảy ra hiện tượng "Ghi đè ngầm" (Silent Override)**.
* **Chi tiết:** Khi notebook khởi chạy trên container của Kaggle:
  1. Hàm [`prepare_resume`](file:///home/u001015/BHSD_SSL/tools/kaggle/resume.py#L5-L77) kết nối vào W&B API và kéo cấu hình gốc `run.config` về:
     ```python
     cfg = copy.deepcopy(dict(run.config))
     ```
  2. File cấu hình thực tế để huấn luyện là `resume_config.yaml` được tạo ra trực tiếp từ `run.config` gốc. Trong file này:
     * `cfg['data']['task']` vẫn là `"multiclass"`
     * `cfg['data']['num_classes']` vẫn là `6`
     * `cfg['data']['class_names']` vẫn là `['Background', 'EDH', 'ICH', 'IVH', 'SAH', 'SDH']`
  3. Cell thực thi của notebook ([`inject_resume`](file:///home/u001015/BHSD_SSL/tools/kaggle/resume.py#L99)) đồng thời gán lại:
     ```python
     TASK = resume_cfg["experiment"]["task"]  # Tự động gán lại thành "multiclass"
     ```
* **Hệ quả:** 
  * Quá trình huấn luyện trên Kaggle vẫn diễn ra bình thường, GPU vẫn chạy và log metrics bình thường.
  * **Tuy nhiên, mô hình vẫn tiếp tục được huấn luyện ở chế độ Multiclass!** Cờ `--task binary` mà người dùng truyền vào ở CLI bị bỏ qua hoàn toàn mà không có bất kỳ cảnh báo nào.

---

### 2.3. Khi nào lỗi Crash thực sự xảy ra?
Hiện tượng **Chặn chủ động** hoặc **Crash kỹ thuật** chỉ xảy ra nếu người dùng cố tình can thiệp thủ công (ví dụ: sửa file `resume_config.yaml` hoặc sửa code để ép `task: binary` nhưng vẫn nạp file trọng số Multiclass):

1. **Chặn chủ động bởi Checkpoint Guard (`validate_checkpoint_context`)**:
   Tại [`methods/base_method.py`](file:///home/u001015/BHSD_SSL/methods/base_method.py#L297), hàm [`validate_checkpoint_context`](file:///home/u001015/BHSD_SSL/utils/checkpoint.py#L214-L227) kiểm tra metadata:
   ```python
   pairs = {
       'task': (task, data.get('task')),                      # 'binary' != 'multiclass'
       'num_classes': (num_classes, data.get('num_classes')),  # 2 != 6
   }
   ```
   Nếu phát hiện lệch, chương trình dừng ngay lập tức:
   ```text
   ValueError: Checkpoint task mismatch: requested binary, saved multiclass
   ```

2. **Lỗi Crash lệch kích thước Tensor (`PyTorch Size Mismatch`)**:
   Nếu vượt qua bước metadata, khi nạp trọng số với cờ `strict=True`, PyTorch sẽ dừng với lỗi:
   ```text
   RuntimeError: Error(s) in loading state_dict for PlainConvUNet:
   size mismatch for seg_outputs.0.weight: copying a param with shape torch.Size([6, 32, 1, 1, 1]) from checkpoint,
   the shape in current model is torch.Size([2, 32, 1, 1, 1]).
   ```

3. **Lệch trạng thái Optimizer & Xung đột W&B**:
   Optimizer momentum tensor của 6 classes không khớp với 2 classes; đồng thời biểu đồ W&B chế độ `same-run` bị gãy schema do xung đột giữa các cột metric `val/dice_EDH...` và `val/dice_Hemorrhage`.

---

## 3. Rủi ro & Tác động Thực tế

1. **Lãng phí Quota GPU Kaggle:** Người dùng tin rằng mình đang tiếp tục huấn luyện mô hình Binary, nhưng GPU lại đang chạy Multiclass (tốn 12h quota hàng tuần mà không đạt đúng mục tiêu thí nghiệm).
2. **Hiểu sai kết quả thí nghiệm:** Khi xem log thấy mô hình vẫn chạy, người dùng có thể nhầm tưởng kiến trúc Binary đã đạt kết quả, trong khi thực tế model đang phân loại 6 class.

---

## 4. Hướng dẫn Khắc phục & Vận hành Chuẩn

### 4.1. Quy tắc vận hành (Dành cho người dùng)
* **Nguyên tắc bất biến của `--resume`:** Cơ chế resume yêu cầu bảo toàn 100% không gian trạng thái (State Space) bao gồm: Task, Architecture, Optimizer, Scheduler và Random State. **Tuyệt đối không đổi `--task` khi gọi `--resume-run`.**
* **Chuyển giao tri thức giữa các bài toán (Transfer Learning / Fine-tuning):**
  * Nếu muốn dùng trọng số của mô hình Multiclass cho bài toán Binary, **không dùng cơ chế resume**.
  * Thay vào đó, cần dùng cơ chế Pretrained Weights Transfer: Chỉ load trọng số Encoder/Backbone, khởi tạo lại Classification Head với `num_classes = 2`, và khởi tạo Optimizer mới từ đầu.

---

### 4.2. Đề xuất Cải tiến Mã nguồn CLI (Code Fix)

Để ngăn chặn triệt để tình trạng Silent Override ở CLI, cần bổ sung bước xác thực sớm (Early Validation) trong [`tools/kaggle/notebook_builder.py`](file:///home/u001015/BHSD_SSL/tools/kaggle/notebook_builder.py).

#### Vị trí sửa đổi đề xuất:
Trong phương thức `build_bundle`:

```python
# tools/kaggle/notebook_builder.py

if resume_run:
    # 1. Truy vấn cấu hình run nguồn từ W&B
    import wandb
    api = wandb.Api()
    run_full_path = resume_run if '/' in resume_run else f"thanhnguyen2005243-ho-chi-minh-city-university-of-technology/BHSD_MeanTeacher/{resume_run}"
    try:
        source_run = api.run(run_full_path)
        source_task = source_run.config.get("data", {}).get("task") or source_run.config.get("experiment", {}).get("task")
        
        # 2. Kiểm tra nếu task chỉ định ở CLI bị lệch với run nguồn
        target_task = os.environ.get("BHSD_TASK")
        if target_task and source_task and target_task != source_task:
            raise ValueError(
                f"[RESUME TASK MISMATCH] Không thể resume run '{resume_run}': "
                f"CLI yêu cầu task='{target_task}', nhưng run nguồn được huấn luyện với task='{source_task}'. "
                f"Quá trình resume bắt buộc phải cùng task và kiến trúc mạng. "
                f"Vui lòng chỉ định '--task {source_task}' hoặc bỏ cờ --task để hệ thống tự động kế thừa."
            )
    except Exception as e:
        print(f"[!] Cảnh báo kiểm tra W&B run: {e}")
```

#### Lợi ích sau khi bổ sung:
* Bắt lỗi ngay tại máy cục bộ (Local), từ chối tạo bundle notebook sai.
* Tiết kiệm thời gian, không gửi nhầm job lên Kaggle và không tiêu tốn GPU quota vô ích.

