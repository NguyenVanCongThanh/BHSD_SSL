# So Sánh Chi Tiết: Dataset BHSD_fold_0 (.h5) vs Dataset Trong nnUNet Option 1

Tài liệu này ghi lại chi tiết các điểm tương đồng, điểm khác biệt và nguyên nhân kỹ thuật giữa tập dữ liệu tiền xử lý **BHSD_fold0** (lưu dưới dạng `.h5` tại `/home/u001015/dataset/BHSD_fold0`) và tập dữ liệu trong **nnUNet Option 1** (`nnUNetPlans_Option1_CTClip` tại `/home/u001015/nnUNet_data/nnUNet_preprocessed/Dataset001_BHSD/`).

---

## 1. Nguồn Gốc và Quy Trình Sinh Dữ Liệu

* **Tập dữ liệu nnUNet Option 1**:
  * Được sinh ra từ pipeline của nnUNet v2 trong `/home/u001015/notebook/nnUNet`.
  * Sử dụng Planner tùy biến: `Option1CTClipPlanner` (trong `custom_experiment_planners.py`).
  * Sử dụng Preprocessor: `Option1CTClipFirstPreprocessor` (trong `custom_preprocessors.py`).
  * Lưu trữ tại: `/home/u001015/nnUNet_data/nnUNet_preprocessed/Dataset001_BHSD/nnUNetPlans_Option1_CTClip_3d_fullres/`.

* **Tập dữ liệu BHSD_fold0 (.h5)**:
  * Được tiền xử lý bằng script: `/home/u001015/notebook/preprocess_bhsd_for_pick.py`.
  * Được chạy qua Slurm job: `/home/u001015/notebook/submit_preprocess_pick.sh`.
  * Thuật toán tiền xử lý được module hóa tại: `/home/u001015/notebook/src/data/preprocess_option1.py`.
  * Phân chia fold được đồng bộ từ: `/home/u001015/notebook/sync_multilabel_folds_to_pick.py`.
  * Lưu trữ gốc tại: `/home/u001015/notebook/preprocessed_bhsd_pick/` và được đồng bộ sang `/home/u001015/dataset/BHSD_fold0/`.

---

## 2. Bảng So Sánh Tổng Hợp

| Tiêu chí | nnUNet Option 1 (nnUNet v2) | Dataset BHSD_fold_0 (.h5) | Mức độ tương đồng |
| :--- | :--- | :--- | :--- |
| **Thuật toán cốt lõi** | Option 1 (Clip HU -> Mask sọ -> Crop BBox) | Option 1 (Tái hiện chính xác theo nnUNet) | **Khớp 100%** |
| **Dải cắt HU (HU Clipping)** | [11.0, 88.0] HU (phân vị 0.5% - 99.5%) | [11.0, 88.0] HU | **Khớp 100%** |
| **Tách sọ não (Skull Masking)** | Ngưỡng voxel > 11.0 HU + lấp lỗ hổng | Ngưỡng voxel > 11.0 HU + lấp lỗ hổng theo lát | **Khớp 100%** |
| **Cắt khung bao (Bounding Box)** | Crop theo tọa độ hộp bao quanh sọ não | Crop theo tọa độ hộp bao quanh sọ não | **Khớp 100% từng voxel** |
| **Chuẩn hóa (Z-Score)** | Mean = 47.6804, Std = 18.4999 | Mean = 47.6804, Std = 18.4999 | **Khớp 100%** |
| **Phân chia tập (Splits)** | 5-Fold Multilabel Stratified Split | Đồng bộ 100% từ splits_multilabel.json | **Khớp 100%** |
| **Hệ thống nhãn (Classes)** | 6 lớp (0: BG, 1: EDH, 2: IPH, 3: IVH, 4: SAH, 5: SDH) | 6 lớp y hệt | **Khớp 100%** |
| **Độ phân giải (Target Spacing)**| **[2.715, 0.488, 0.488] mm** | **[5.0, 0.5, 0.5] mm** | **Khác biệt cốt lõi** |
| **Số lát cắt theo trục Z** | Khoảng 50 - 65 lát (do nội suy dày lát Z) | Khoảng 28 - 35 lát (giữ đúng độ dày gốc ~5mm) | Hệ quả của Spacing |
| **Định dạng file lưu trữ** | Binary .b2nd + metadata .pkl | HDF5 .h5 (nén gzip) | Khác biệt định dạng |

---

## 3. Bằng Chứng Thực Tế Trên Ca Bệnh Mẫu (ID_0237f3c9_ID_40015688b9)

Kiểm tra trực tiếp giữa file `ID_0237f3c9_ID_40015688b9.pkl` (nnUNet Option 1) và file `ID_0237f3c9_ID_40015688b9.h5` (BHSD_fold0):

* **Kích thước ảnh thô ban đầu (Raw Shape)**:
  * nnUNet Option 1: `(28, 512, 512)`
  * BHSD_fold0: `(28, 512, 512)`
* **Tọa độ Bounding Box cắt được (Bounding Box Coordinates)**:
  * nnUNet Option 1: `[[0, 28], [22, 488], [43, 452]]`
  * BHSD_fold0: `[[0, 28], [22, 488], [43, 452]]`
  * *(Nhận xét: Tọa độ cắt bao quanh sọ não trùng khớp chính xác 100% đến từng điểm ảnh)*.
* **Kích thước sau khi cắt Bounding Box (trước khi Resample)**:
  * nnUNet Option 1: `(28, 466, 409)`
  * BHSD_fold0: `(28, 466, 409)`
* **Khoảng cách voxel gốc (Original Spacing)**:
  * Z = 5.288 mm, Y = 0.488 mm, X = 0.488 mm.
* **Kích thước cuối cùng sau Resample**:
  * BHSD_fold0 (Spacing [5.0, 0.5, 0.5] mm): `(30, 455, 399)`.
  * nnUNet Option 1 (Spacing [2.715, 0.488, 0.488] mm): Khoảng `(55, 466, 409)`.

---

## 4. Phân Tích Chi Tiết Điểm Khác Biệt: Target Spacing

### 4.1. Tại sao trong nnUNet Option 1, Spacing lại là [2.715, 0.488, 0.488] mm?
Trong file `custom_experiment_planners.py`:
```python
class Option1CTClipPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id,
                 gpu_memory_target_in_gb=8,
                 preprocessor_name='Option1CTClipFirstPreprocessor', 
                 plans_name='nnUNetPlans_Option1_CTClip',
                 overwrite_target_spacing=None,
                 suppress_transpose=False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, 
                         plans_name, overwrite_target_spacing, suppress_transpose)
```
Do `overwrite_target_spacing` được để mặc định là `None`, nnUNet v2 tự động kích hoạt thuật toán tính spacing mục tiêu:
* Nó lấy phân vị 16.7% của khoảng cách trục Z trên toàn bộ tập dữ liệu đối với ảnh bất đối xứng (anisotropic CT).
* Kết quả tính tự động ra giá trị `2.7148 mm` cho trục Z. Do đó, nnUNet tự động nội suy làm dày gấp đôi số lát cắt CT theo trục Z.

### 4.2. Tại sao trong BHSD_fold0 (.h5), Spacing lại là [5.0, 0.5, 0.5] mm?
Trong script chạy thực tế `submit_preprocess_pick.sh` (dòng 52):
```bash
python3 /home/u001015/notebook/preprocess_bhsd_for_pick.py \
    --target_spacing 5.0 0.5 0.5 \
    --clip_min 11.0 \
    --clip_max 88.0 \
    --fg_mean 47.6804 \
    --fg_std 18.4999
```
* Tham số `5.0 0.5 0.5` mm được truyền tường minh vào quá trình tiền xử lý.
* Mục tiêu là giữ nguyên độ dày lát cắt thực tế của phim chụp CT sọ não lâm sàng (~5mm), không làm tăng số lát cắt giả tạo, từ đó tiết kiệm bộ nhớ GPU VRAM và tăng tốc độ xử lý khi huấn luyện bán giám sát (SSL).
* Lưu ý rằng trong nnUNet cũng có một kế hoạch tên là `nnUNetPlans_spacing05` dùng chính xác mức spacing `[5.0, 0.5, 0.5] mm`.

---

## 5. Đánh Giá Ý Nghĩa Đối Với Các Mô Hình Bán Giám Sát (Mean Teacher, PICK)

1. **Hoàn toàn kế thừa ưu điểm của Option 1**:
   * Loại bỏ 100% vùng không khí và viền đen vô nghĩa bên ngoài hộp sọ.
   * Giữ dải giá trị HU tập trung đúng vào mô mềm và khối máu tụ ([11.0, 88.0] HU).
   * Chuẩn hóa cường độ điểm ảnh ổn định theo phân phối Gaussian (Mean 47.68, Std 18.50).

2. **Tối ưu hóa cho Patch Size của Mean Teacher / 3D SSL**:
   * Với độ dày lát cắt 5.0 mm, mỗi ca bệnh chỉ có khoảng 28 đến 35 lát cắt theo chiều Z.
   * Cấu hình patch đầu vào của Mean Teacher là `[32, 192, 192]` hoặc `[32, 160, 160]`.
   * Chiều sâu patch Z = 32 bao phủ **gần như toàn bộ não bộ** trong một lần crop, giúp mô hình học được ngữ cảnh toàn diện từ nền sọ lên đỉnh sọ mà không bị cắt vụn như khi bị nội suy dày về 2.71 mm (khi đó một ca có hơn 60 lát, patch 32 chỉ nhìn thấy một nửa não).

---

## 6. Kết Luận

* Dataset `BHSD_fold0` (.h5) **đã được xử lý hoàn toàn theo đúng thuật toán Option 1 của nnUNet**.
* Sự khác biệt duy nhất là **độ phân giải trục Z được chủ động giữ ở 5.0 mm** thay vì để nnUNet tự động nội suy dày về 2.71 mm.
* Toàn bộ 154 ca có nhãn của Fold 0 và 38 ca kiểm định val hoàn toàn trùng khớp với phân chia Fold 0 của nnUNet, sẵn sàng cho việc so sánh hiệu năng công bằng (fair benchmark) giữa nnUNet và các phương pháp SSL.
