# Hướng Dẫn Chuẩn Hóa Benchmark: Mean Teacher vs nnUNet (MedSSL-BHSD)

Tài liệu này tổng hợp toàn diện các điểm tương đồng, điểm khác biệt và lộ trình kỹ thuật cần thực hiện nhằm đưa mô hình **Mean Teacher** (trong codebase `BHSD_SSL`) về cùng một mặt bằng so sánh (fair benchmark) với **nnUNet v2** (đặc biệt là lần chạy Option 1 và Spacing 0.5).

---

## 1. Bảng So Sánh Tổng Thể Toàn Diện

| Hạng mục so sánh | nnUNet Option 1 (nnUNet v2) | Mean Teacher (Codebase Đã Chuẩn Hóa) | Mức độ ảnh hưởng đến Benchmark |
| :--- | :--- | :--- | :--- |
| **Dữ liệu: Ngưỡng HU** | [11.0, 88.0] HU | [11.0, 88.0] HU | Đồng nhất (Khớp 100%) |
| **Dữ liệu: Crop Hộp sọ** | Lọc sọ não + Crop Bounding Box | Lọc sọ não + Crop Bounding Box | Đồng nhất (Khớp 100% tọa độ) |
| **Dữ liệu: Phân chia Fold** | 5-Fold Multilabel Stratified Split | Đồng bộ từ splits_multilabel.json | Đồng nhất (Khớp 100% danh sách ca) |
| **Dữ liệu: Target Spacing** | [2.715, 0.488, 0.488] mm | [5.0, 0.5, 0.5] mm | **Rất cao (Lệch độ phân giải Z)** |
| **Không gian tính Metric** | Không gian ảnh gốc NIfTI (.nii.gz) | Không gian tiền xử lý (.h5) | **Rất cao (Lệch thước đo thực tế)** |
| **Kỹ thuật Sliding Window** | Gaussian Importance Weighting | Gaussian Importance Weighting (Đã xong) | **Đã đồng nhất (Khớp 100%)** |
| **Test-Time Augmentation** | Mặc định bật TTA lật 3 trục (8 lần) | Hỗ trợ cờ `--use_tta` lật 3 trục (Đã xong) | **Đã đồng nhất (Khớp 100%)** |
| **Số tầng mạng (Stages)** | 7 stages ([32, 64, 128, 256, 320, 320, 320]) | 7 stages chuẩn nnUNet v2 (Đã xong) | **Đã đồng nhất (Khớp 100%)** |
| **Nhân tích chập (Kernel)** | 1x3x3 ở 2 tầng đầu, 3x3x3 ở tầng sau | (1, 3, 3) ở tầng đầu, (3, 3, 3) sau (Đã xong) | **Đã đồng nhất (Khớp 100%)** |
| **Cơ chế Downsampling** | Strided Convolution 3D có học | nn.Conv3d bước nhảy có học (Đã xong) | **Đã đồng nhất (Khớp 100%)** |
| **Data Augmentation** | Rất mạnh (Scale, Elastic, Blur, Gamma) | Đã thêm Scaling3D, Gamma, Blur3D (Đã xong) | **Đã nâng cấp tương đương** |
| **Ngân sách huấn luyện** | 1,000 epochs = 250,000 steps (~44h) | 12,000 steps (~48 epochs của nnUNet) | **Rất cao (Chưa khai thác hết tiềm năng)** |

---

## 2. Phân Tích 5 Điểm Sai Lệch Trọng Yếu Cần Chỉnh Sửa

### 2.1. Độ Lệch Không Gian Dữ Liệu và Metric Đánh Giá
* **Vấn đề**:
  1. nnUNet Option 1 tự động nội suy làm mịn lát cắt Z xuống 2.71 mm, làm tăng gấp đôi số lát cắt. Mean Teacher trong `BHSD_fold0` (.h5) giữ nguyên độ dày thực tế 5.0 mm.
  2. nnUNet đánh giá kết quả bằng cách resample ngược lại kích thước và spacing gốc của file NIfTI (.nii.gz) của bệnh nhân rồi mới tính Dice so với Ground Truth gốc. Mean Teacher hiện tại tính Dice trực tiếp trên ma trận `.h5` đã tiền xử lý.
* **Tác động**: Nếu tính Dice trên hai không gian voxel khác nhau, kết quả so sánh sẽ bị sai lệch nghiêm trọng và thiếu cơ sở khoa học.

### 2.2. Cơ Chế Ghép Mảnh Cửa Sổ Trượt (Sliding Window Aggregation)
* **Vấn đề**: Khi cửa sổ trượt dịch chuyển, các patch sẽ đè lên nhau. nnUNet dùng ma trận Gaussian (tâm patch có trọng số 1.0, mép biên giảm dần về 0.0) giúp vùng tiếp giáp giữa các patch mượt mà, không bị sọc biên. Mean Teacher hiện tại dùng trung bình đều (`score_map / cnt`), khiến vùng tiếp giáp biên giữa các patch dễ bị gián đoạn đặc trưng.
* **Tác động**: Khiến Mean Teacher mất khoảng 0.5% - 1.5% mDice do lỗi răng cưa ở biên của các patch.

### 2.3. Thiếu Hụt Test-Time Augmentation (TTA)
* **Vấn đề**: nnUNet khi chạy lệnh dự đoán mặc định lật thể tích qua các trục X, Y, Z (8 tổ hợp lật), dự đoán rồi lật ngược lại và lấy trung bình xác suất. Mean Teacher chỉ chạy một lượt duy nhất.
* **Tác động**: TTA giúp mô hình ổn định hơn và thường đóng góp từ 1.0% đến 2.0% Dice trên các cấu trúc xuất huyết mỏng hoặc rải rác (như SAH, SDH).

### 2.4. Bất Đối Xứng Về Data Augmentation Khi Huấn Luyện
* **Vấn đề**:
  * nnUNet áp dụng: Co dãn ngẫu nhiên 3D (Scaling từ 0.7x đến 1.4x), xoay đa trục góc tự do, làm mờ Gaussian Blur, mô phỏng ảnh chất lượng thấp (Simulate Low Resolution), biến đổi Gamma.
  * Mean Teacher (`data/transforms3d.py`): Mới chỉ hỗ trợ lật/xoay 90 độ (`RandomRotFlip`) và cắt ngẫu nhiên có ưu tiên vùng tổn thương (`RandomCrop`).
* **Tác động**: Thiếu co dãn kích thước và biến đổi độ tương phản khiến mô hình học vẹt (overfitting) trên tập nhãn hạn chế, giảm khả năng tổng quát hóa trên tập kiểm định.

### 2.5. Bất Đối Xứng Về Ngân Sách Huấn Luyện (Training Budget)
* **Vấn đề**: nnUNet Option 1 được huấn luyện đầy đủ 1,000 epochs (tương đương 250,000 iterations). Mean Teacher hiện tại cấu hình `max_iteration: 12000` (ít hơn hơn 20 lần).
* **Tác động**: Mạng Teacher cần hàng vạn bước để trọng số trung bình động (EMA decay 0.999) tích lũy đủ tri thức sâu và làm mượt gradient từ dữ liệu không nhãn. Dừng lại ở 12,000 bước là quá sớm để đạt đỉnh hiệu năng.

---

## 3. Lộ Trình Kỹ Thuật và Các Chỉnh Sửa Cần Thực Hiện

### Giai Đoạn 1: Chuẩn Hóa Không Gian Đánh Giá (Evaluation Space Alignment)
Cần bổ sung một module xuất kết quả phân đoạn chuẩn y khoa trong `evaluate.py`:
1. Nạp file ảnh NIfTI gốc (`.nii.gz`) và file nhãn gốc từ thư mục gốc của BHSD.
2. Dùng mô hình Teacher dự đoán trên thể tích `.h5`.
3. Sử dụng phép nội suy láng giềng gần nhất (Nearest Neighbor) để đưa ma trận nhãn dự đoán về đúng kích thước và spacing gốc của ca bệnh.
4. Tính toán Dice Score và Hausdorff Distance 95 (HD95) trên không gian NIfTI gốc này để đối chiếu trực tiếp với kết quả của nnUNet.

### Giai Đoạn 2: Nâng Cấp Bộ Suy Luận Cửa Sổ Trượt (Inference Engine)
Thực hiện chỉnh sửa trực tiếp trong file `evaluation/sliding_window.py`:
1. **Tạo ma trận Gaussian Weighting**: Tạo ma trận kích thước bằng patch (ví dụ 32 x 192 x 192) với phân phối chuẩn nhiều chiều đạt đỉnh ở tâm và tiệm cận 0 ở biên; nhân ma trận này với xác suất softmax của từng patch trước khi cộng dồn.
2. **Tích hợp tùy chọn Test-Time Augmentation (TTA)**: Bổ sung cờ `--use_tta` để tự động lật 3 chiều không gian khi suy luận trên tập kiểm định, hoặc khi chạy so sánh với nnUNet cần truyền cờ `--disable_tta` cho nnUNet để cả hai bên đều không dùng TTA.

### Giai Đoạn 3: Bổ Sung Data Augmentation Vào Codebase MedSSL-BHSD
Mở rộng file `data/transforms3d.py` với các lớp biến đổi sau:
1. `RandomScaling3D`: Co dãn thể tích ngẫu nhiên trong khoảng [0.7, 1.4] trước khi cắt patch.
2. `GaussianBlur3D`: Làm mờ nhẹ ngẫu nhiên với xác suất 20% (sigma từ 0.5 đến 1.0) nhằm mô phỏng độ nhiễu khác nhau giữa các máy quét CT.
3. `GammaTransform`: Biến đổi hàm mũ gamma trong khoảng [0.7, 1.5] để làm phong phú độ tương phản mô máu tụ.

### Giai Đoạn 4: Tinh Chỉnh Kiến Trúc Backbone `nnUNet3D` (Đã Hoàn Thành)
Trong file `models/backbones/nnunet3d.py`:
1. **Hỗ trợ Kernel bất đối xứng (`anisotropic_kernels: true`)**: Do lát cắt Z dày ($5.0\text{ mm}$ so với $0.5\text{ mm}$ ở mặt phẳng axial), các tầng `enc1`, `enc2` và `dec1`, `dec2` sử dụng kernel phẳng `(1, 3, 3)` và stride `(1, 2, 2)` nhằm bảo toàn chi tiết Z và tránh mờ thông tin giữa các lát cắt dày.
2. **Cơ chế Strided Convolution (`use_strided_conv: true`)**: Thay thế `nn.MaxPool3d` bằng tích chập có bước nhảy có thể học (`nn.Conv3d(stride=stride)`), bám sát 100% thiết kế của nnUNet v2.
3. **Patch Size huấn luyện**: Thiết lập chuẩn `[32, 192, 192]` (bao quát trọn vẹn não trên mặt phẳng axial $192 \times 192$ và toàn bộ $28 - 32$ lát cắt Z), hoàn toàn tương thích và cân bằng với patch size `[28, 192, 192]` của `nnUNetPlans_spacing05`.

### Giai Đoạn 5: Thiết Lập Ngân Sách Huấn Luyện và Giao Thức Benchmark Chuẩn
* **Ngân sách**: Tăng `max_iteration` trong `configs/methods/mean_teacher.yaml` từ `12000` lên tối thiểu **`30000` đến `50000` steps**, điều chỉnh lịch trình EMA cosine rampdown phù hợp tương ứng.
* **Giao thức so sánh khoa học**:
  * **Kịch bản A (So sánh với Spacing 5.0mm)**: So sánh trực tiếp kết quả của Mean Teacher (dùng `BHSD_fold0` spacing 5.0mm) với kết quả chạy của **`nnUNetPlans_spacing05`** (cũng chạy trên spacing 5.0mm). Đây là cách làm nhanh và công bằng nhất mà không cần phải tiền xử lý lại toàn bộ 2,170 ca bệnh.
  * **Kịch bản B (So sánh với Option 1 Spacing 2.71mm)**: Cần chạy lại script tiền xử lý batch với `--target_spacing 2.7148 0.4883 0.4883` để đồng nhất độ phân giải trước khi huấn luyện Mean Teacher.

---

## 4. Bảng Kế Hoạch Triển Khai Cụ Thể (Action Checklist)

- [x] **Bước 1**: Thêm ma trận trọng số Gaussian vào hàm `sliding_window_inference_3d` trong `evaluation/sliding_window.py` (Đã hoàn thành: bổ sung hàm `compute_gaussian_weight_map` với bộ nhớ đệm `@lru_cache`, tích hợp vào `sliding_window_inference_3d`, `evaluate_dataset_sliding_window`, `benchmark_5folds` và hỗ trợ cờ `--no_gaussian` trong `evaluate.py`).
- [x] **Bước 2**: Thêm cờ `--use_tta` và logic lật 3 chiều vào bộ suy luận đánh giá (Đã hoàn thành: bổ sung hàm `_forward_with_tta` với 7 tổ hợp lật 3D không gian, tích hợp vào `sliding_window_inference_3d`, `evaluate_dataset_sliding_window`, `benchmark_5folds` và cờ `--use_tta` trong `evaluate.py`).
- [ ] **Bước 3**: Bổ sung hàm xuất mask dự đoán ra `.nii.gz` và tính Dice trên không gian NIfTI gốc (Tạm hoãn theo yêu cầu).
- [x] **Bước 4**: Thêm `RandomScaling3D`, `GammaTransform`, và `GaussianBlur3D` vào `data/transforms3d.py` (Đã hoàn thành: triển khai đầy đủ 3 lớp biến đổi dữ liệu 3D chuẩn nnUNet, xuất qua `data/__init__.py` và tích hợp tự động vào pipeline `train_transforms` trong `methods/base_method.py`).
- [x] **Bước 4.1**: Chuẩn hóa Kernel các tầng đầu, 7 stages và Downsampling có học trong `models/backbones/nnunet3d.py` (Đã hoàn thành: nâng cấp lên 7 stages chuẩn nnUNet v2 Option 1, hỗ trợ `anisotropic_kernels: true` với kernel `(1, 3, 3)` ở tầng đầu, `use_strided_conv: true` thay thế MaxPool3d, tương thích patch size `[32, 192, 192]` và `[28, 192, 192]`).
- [ ] **Bước 5**: Tăng `max_iteration` lên 30,000 - 50,000 steps trong cấu hình huấn luyện.
- [ ] **Bước 6**: Lựa chọn đối chiếu với `nnUNetPlans_spacing05` (nếu giữ spacing 5.0mm) hoặc resample lại theo Option 1 (nếu giữ spacing 2.71mm).

---

## 5. Hướng Dẫn Sử Dụng Chi Tiết Các Cờ Khi Chạy (Execution & Flags Guide)

Phần này ghi rõ cơ chế áp dụng (tự động hay thủ công) và các câu lệnh mẫu tương ứng:

### 5.1. Bảng Tóm Tắt Cơ Chế Áp Dụng Của Từng Hạng Mục

| Tính năng / Module | Cơ chế áp dụng | Có cần cờ CLI không? | Ghi chú & Giá trị mặc định |
| :--- | :--- | :--- | :--- |
| **Kiến trúc mạng 7 Stages** | **Tự động 100%** | **Không cần** | Cấu hình trong `configs/methods/mean_teacher.yaml` (`num_stages: 7`). |
| **Kernel Anisotropic (1, 3, 3)** | **Tự động 100%** | **Không cần** | Cấu hình `anisotropic_kernels: true` trong yaml/backbone. |
| **Strided Conv Downsampling** | **Tự động 100%** | **Không cần** | Cấu hình `use_strided_conv: true` thay thế hoàn toàn MaxPool3d. |
| **Data Augmentation Mới** | **Tự động 100%** | **Không cần** | Tự động kích hoạt trong pipeline DataLoader (`Scaling3D`, `Gamma`, `Blur3D`). |
| **Gaussian Sliding Window** | **Mặc định BẬT (100%)** | **Không cần** (Trừ khi tắt) | Mặc định áp dụng cho cả validation khi train lẫn khi chạy `evaluate.py`. Nếu muốn tắt để đối chứng, thêm `--no_gaussian`. |
| **Test-Time Augmentation (TTA)** | **Chỉ định qua cờ** | **CẦN THÊM CỜ `--use_tta`** | Mặc định TẮT khi train để không làm chậm validation. Khi benchmark đối đầu với nnUNet, **bắt buộc thêm `--use_tta`**. |

---

### 5.2. Các Câu Lệnh Mẫu Chuẩn Khi Thực Thi

#### 1. Huấn Luyện (Training)
Tất cả kiến trúc 7 tầng, kernel `(1, 3, 3)`, giảm mẫu có học, Gaussian validation, và Data Augmentation mở rộng đều **hoàn toàn tự động**:
```bash
python train.py --config configs/methods/mean_teacher.yaml
```

#### 2. Đánh Giá Suy Luận Thông Thường (Standard Inference - Nhanh)
Mặc định đã **bật sẵn Trọng số Gaussian**:
```bash
# Đánh giá 1 checkpoint cụ thể:
python evaluate.py --checkpoint /path/to/best_model.pth --fold 0

# Benchmark tổng thể 5-fold (OOF):
python evaluate.py --exp_dir /home/u001015/experiments/BHSD_mean_teacher_5folds
```

#### 3. Đánh Giá Đối Đầu Sòng Phẳng Với nnUNet v2 (Fair Benchmark Inference)
nnUNet v2 mặc định bật TTA lật 3 chiều không gian (8 lần suy luận). Để đối đầu công bằng 100%, bạn **thêm cờ `--use_tta`**:
```bash
# Đánh giá 1 checkpoint có đầy đủ Gaussian + TTA (8 biến thể lật):
python evaluate.py --checkpoint /path/to/best_model.pth --fold 0 --use_tta

# Benchmark tổng thể 5-fold đầy đủ Gaussian + TTA:
python evaluate.py --exp_dir /home/u001015/experiments/BHSD_mean_teacher_5folds --use_tta
```

#### 4. Đánh Giá Thử Nghiệm Đối Chứng (Ablation Study)
Nếu bạn muốn đo lường chính xác xem **Trọng số Gaussian đóng góp bao nhiêu % Dice**, hãy chạy với cờ `--no_gaussian`:
```bash
# Đánh giá không dùng Gaussian (trung bình đều phẳng cũ):
python evaluate.py --checkpoint /path/to/best_model.pth --fold 0 --no_gaussian
```
