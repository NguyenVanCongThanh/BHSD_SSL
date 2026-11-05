# Báo Cáo Thiết Kế Kiến Trúc & Đối Chiếu Siêu Tham Số: 7-Stage Residual V-Net vs. nnU-Net 3D

> **Tài liệu đặc tả kỹ thuật thực nghiệm bán giám sát BHSD (Binary Task - Fold 0 - 5% Labels)**  
> **Tác giả:** AI Assistant & Nghiên cứu viên  
> **Ngày cập nhật:** 05/11/2026  
> **Cấu hình thực nghiệm:** [configs/experiments/bhsd_mt_vnet_binary_fold0.yaml](file:///home/u001015/BHSD_SSL/configs/experiments/bhsd_mt_vnet_binary_fold0.yaml)  
> **Baseline đối chiếu:** Run `y3jf7hjh` trên Weights & Biases (`BHSD_binary_MeanTeacher_fold0_20261102_085643_mean_teacher_fold0`)

---

## 1. Mục Đích & Nguyên Tắc Khoa Học (Controlled Ablation Benchmark)

Nghiên cứu này nhằm mục đích so sánh công bằng hiệu quả phân đoạn xuất huyết não 3D (BHSD) giữa hai trường phái mạng nơ-ron tích chập kinh điển trong y tế:
1. **Plain CNN (nnU-Net 3D)**: Không có đường truyền tắt nội khối (No residual shortcuts).
2. **Residual CNN (V-Net)**: Có đường truyền tắt phần dư nội khối (Residual shortcuts) theo đề xuất gốc của Milletari et al. (3DV 2016).

### Nguyên tắc kiểm soát biến số (Isolation of Variables):
- **Biến độc lập duy nhất (Independent Variable):** Cơ chế cộng phần dư nội tầng (*Residual Block* vs. *Plain Block*).
- **Các biến kiểm soát giữ nguyên 100% (Controlled Variables):** 
  - Tập dữ liệu, Random seed, 8 ca bệnh có nhãn cố định (`stratified` sampling).
  - Dung lượng mô hình (Model capacity ~46M tham số).
  - Độ sâu mạng (7 stages), số kênh đặc trưng (`[32, 64, 128, 256, 320, 320, 320]`).
  - Lộ trình lấy mẫu không gian (Downsampling stride schedule và anisotropic kernels).
  - Đa tỉ lệ giám sát sâu (Deep Supervision: 1 Main + 5 Aux Heads).
  - Tất cả các siêu tham số tối ưu hóa và bán giám sát (Mean Teacher consistency, EMA schedule, Poly LR, AMP FP16).

---

## 2. Bảng Đối Chiếu Chi Tiết Toàn Diện 1-1

| Hạng mục | Tham số / Đặc tính kỹ thuật | nnU-Net 3D (Baseline `y3jf7hjh`) | 7-Stage V-Net (Đề xuất mới) | Ý nghĩa & Lý do khoa học |
| :--- | :--- | :--- | :--- | :--- |
| **I. Tác vụ & Dữ liệu** | Task | `binary` (2 classes: Background, ICH) | `binary` (2 classes: Background, ICH) | Đồng nhất bài toán phân loại nhị phân |
| | Fold Cross-Validation | Fold 0 | Fold 0 | Cùng tập phân chia kiểm định |
| | Ngẫu nhiên & Tái lập | `seed: 1337`, `deterministic: True` | `seed: 1337`, `deterministic: True` | Cố định điều kiện ngẫu nhiên |
| | Giới hạn Pool huấn luyện | 154 ca (Tập sạch đã kiểm duyệt) | 154 ca (Tập sạch đã kiểm duyệt) | Loại bỏ 1,978 ca chưa lọc |
| | Số ca có nhãn (`labelnum`) | 8 ca (tương đương 5% của 154) | 8 ca (tương đương 5% của 154) | Cùng tỉ lệ giám sát cực thấp |
| | Danh sách ca có nhãn | `[3, 27, 57, 68, 85, 98, 114, 123]` | `[3, 27, 57, 68, 85, 98, 114, 123]` | Khóa 100% ID các ca có nhãn |
| | Kích thước Patch đầu vào | `[28, 256, 256]` ($Z \times Y \times X$) | `[28, 256, 256]` ($Z \times Y \times X$) | Cùng không gian thụ cảm đầu vào |
| | Data Augmentation | 3D Scaling, Gamma, Gaussian Blur | 3D Scaling, Gamma, Gaussian Blur | Cùng chính sách điều hòa dữ liệu |
| **II. Kiến trúc Mô hình** | **Đường truyền nội khối** | **Plain (Không có shortcut)** | **Residual: $y = \sigma(f(x) + W_s x)$** | **Khảo sát tác động của Residual Link** |
| | **Long Skip (Enc $\to$ Dec)** | **Concat (`torch.cat`)** | **Concat (`torch.cat`)** | **Chuẩn xác theo paper gốc Milletari 2016** |
| | Số mức phân giải (Stages) | 7 stages (Enc 1-6, Bottleneck, Dec 6-1) | 7 stages (Enc 1-6, Bottleneck, Dec 6-1) | Đồng nhất độ sâu và Receptive Field |
| | Số Conv mỗi block | 2 Convs | 2 Convs | Giữ số phép tính/tham số tương đồng |
| | Feature Channels | `[32, 64, 128, 256, 320, 320, 320]` | `[32, 64, 128, 256, 320, 320, 320]` | Cùng độ rộng biểu diễn đặc trưng |
| | Kích thước Bottleneck | 320 channels, spatial `[7, 4, 4]` | 320 channels, spatial `[7, 4, 4]` | Cùng độ nén thông tin tại đáy mạng |
| | Kernel bất đẳng hướng | Stage 1-2: `(1,3,3)`; Stage 3-7: `(3,3,3)`| Stage 1-2: `(1,3,3)`; Stage 3-7: `(3,3,3)`| Bảo toàn độ dày lát cắt CT theo trục Z |
| | Lộ trình Strides Downsample| `[(1,2,2), (1,2,2), (2,2,2), (2,2,2), (1,2,2), (1,2,2)]` | `[(1,2,2), (1,2,2), (2,2,2), (2,2,2), (1,2,2), (1,2,2)]` | Trục Z chỉ chia 4 ($28 / 4 = 7$), không lỗi tensor |
| | Cơ chế lấy mẫu giảm | Strided Conv3D (không dùng MaxPool) | Strided Conv3D (không dùng MaxPool) | Học trọng số downsampling |
| | Cơ chế lấy mẫu tăng | ConvTranspose3D | ConvTranspose3D | Học trọng số upsampling |
| | Chuẩn hóa (Normalization) | `InstanceNorm3d(affine=True)` | `InstanceNorm3d(affine=True)` | Khử phụ thuộc thống kê batch size nhỏ |
| | Hàm kích hoạt (Activation) | `LeakyReLU(negative_slope=0.01)` | `LeakyReLU(negative_slope=0.01)` | Khắc phục hiện tượng Dead Neurons |
| | Dropout | Tắt (`has_dropout=False`) | Tắt (`has_dropout=False`) | Đồng nhất quy tắc chính quy hóa |
| | **Tổng số tham số (Params)** | **46,067,276 (~46.07M)** | **46,779,756 (~46.78M)** | **Chênh lệch chỉ ~1.5% do 1x1 shortcut conv** |
| **III. Deep Supervision** | Số lượng nhánh giám sát | 1 Main Head + 5 Auxiliary Heads | 1 Main Head + 5 Auxiliary Heads | Giám sát gradient đa tỉ lệ |
| | Vị trí trích xuất Aux Heads | Decoder 6, 5, 4, 3, 2 | Decoder 6, 5, 4, 3, 2 | Cùng các cấp độ phân giải trung gian |
| | Cấu trúc mỗi Head | `Conv3d(in_ch, 2, kernel_size=1)` | `Conv3d(in_ch, 2, kernel_size=1)` | Cùng phép chiếu phân lớp 1x1 |
| | Trọng số mất mát Aux | `[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]` | `[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]` | Phân rã gradient từ coarse đến fine |
| **IV. Bán Giám Sát (SSL)** | Phương pháp | Mean Teacher (Tarvainen & Valpola) | Mean Teacher (Tarvainen & Valpola) | Cùng khung SSL nhất quán theo thời gian |
| | Kích thước Batch | 8 mẫu (DDP 4 mẫu / GPU trên 2x T4) | 8 mẫu (DDP 4 mẫu / GPU trên 2x T4) | Cùng số lượng mẫu cập nhật gradient |
| | Tỉ lệ Labeled / Unlabeled | 4 mẫu có nhãn / 4 mẫu không nhãn | 4 mẫu có nhãn / 4 mẫu không nhãn | Cùng cân bằng thông tin giám sát |
| | Hàm mất mát có giám sát | SoftDiceLoss + CrossEntropyLoss | SoftDiceLoss + CrossEntropyLoss | Cùng hàm mục tiêu phân đoạn |
| | Hàm mất mát nhất quán | `hybrid` (Class-balanced channel MSE) | `hybrid` (Class-balanced channel MSE) | Cùng phương pháp tính consistency |
| | Trọng số Background | 0.05 (Downweight background) | 0.05 (Downweight background) | Chống hiện tượng sụp đổ mô hình (collapse) |
| | Trọng số nhất quán tối đa | 0.05 | 0.05 | Cùng mức độ tác động của consistency loss |
| | Lộ trình Ramp-up | 4,000 steps (Sigmoid ramp-up) | 4,000 steps (Sigmoid ramp-up) | Cùng tiến độ tăng dần độ tin cậy |
| | Lộ trình Ramp-down | Bắt đầu từ 7,800 steps, min factor 0.25 | Bắt đầu từ 7,800 steps, min factor 0.25 | Ổn định trọng số giai đoạn cuối |
| | Teacher EMA Decay | 0.999 (Cosine ramp-up từ 0.99) | 0.999 (Cosine ramp-up từ 0.99) | Cùng tốc độ làm mượt trọng số Teacher |
| | Nhiễu Student (Perturbation)| Gaussian Noise (`std=0.02`), Intensity (`0.05`)| Gaussian Noise (`std=0.02`), Intensity (`0.05`)| Cùng mức độ nhiễu thách thức Student |
| **V. Tối Ưu Hóa & Đánh Giá**| Thuật toán tối ưu | SGD (Momentum = 0.9) | SGD (Momentum = 0.9) | Cùng optimizer |
| | Learning Rate cơ sở | 0.01 | 0.01 | Cùng tốc độ học ban đầu |
| | Weight Decay | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | Cùng hệ số phạt trọng số L2 |
| | Lộ trình suy giảm LR | `poly` ($power = 0.7$) | `poly` ($power = 0.7$) | Cùng quy luật giảm tốc độ học |
| | Tổng số bước huấn luyện | 12,000 steps | 12,000 steps | Cùng thời lượng huấn luyện |
| | Chu kỳ Validation trượt | Mỗi 200 steps trên Teacher | Mỗi 200 steps trên Teacher | Cùng tần suất kiểm tra chất lượng |
| | Bước trượt Sliding Window | Stride XY = 32, Stride Z = 8 | Stride XY = 32, Stride Z = 8 | Cùng mật độ inference trượt 3D |
| | Trọng số Gaussian viền | Bật (`val_gaussian: True`) | Bật (`val_gaussian: True`) | Khử nhiễu không liên tục giữa các patch |
| | Bộ nhớ & Tính toán | AMP FP16 (`GradScaler`) | AMP FP16 (`GradScaler`) | Cùng tối ưu hóa Tensor Cores trên GPU |

---

## 3. Kiến Trúc Khối Chi Tiết: `VNetResidualBlock`

Khối cơ bản của V-Net 7 tầng được thiết kế nhằm duy trì biểu thức chuẩn của khối Residual kết hợp chuẩn hóa và kích hoạt tối ưu cho dữ liệu y tế 3D:

```
           Input x (C_in, D, H, W)
                |---------------------\
                |                      |  [1x1 Conv + InstanceNorm3d]
          Conv3d (k)                   |  (nếu C_in != C_out)
                |                      |
          InstanceNorm3d               |
                |                      |
          LeakyReLU (0.01)             |
                |                      |
          Conv3d (k)                   |
                |                      |
          InstanceNorm3d               |
                |                      |
                v                      v
             F(x)  +  Shortcut(x) <---/
                |
          LeakyReLU (0.01)
                |
           Output y (C_out, D, H, W)
```

---

## 4. Bảo Vệ Tính Toàn Vẹn Mã Nguồn (Codebase Integrity)

- **[models/backbones/nnunet3d.py](file:///home/u001015/BHSD_SSL/models/backbones/nnunet3d.py)**: Giữ nguyên bản 100% (**Zero-diff** so với commit gốc). Mọi thực nghiệm trước đó sử dụng `nnunet3d` và việc nạp lại checkpoint `best_model.pth` từ run `y3jf7hjh` đều được bảo toàn tính tương thích hoàn hảo.
- **[models/backbones/vnet.py](file:///home/u001015/BHSD_SSL/models/backbones/vnet.py)**: Độc lập hoàn toàn, chứa class `VNet7Stage` và `VNetResidualBlock`, đồng thời hỗ trợ chuyển mạch thông minh (`num_stages=5` cho V-Net cổ điển và `num_stages=7` cho V-Net hiện đại).

---

## 5. Hướng Dẫn Thực Thi

### Lệnh 1: Submit lên Kaggle Dual T4 (Khuyên dùng)
```bash
bash scripts/submit_mean_teacher_kaggle.sh \
  --task binary \
  --setting vnet \
  --labelnum 8 \
  --max-samples 154 \
  --labeled-sampling stratified
```

### Lệnh 2: Chạy trực tiếp trên máy chủ đa GPU (DDP qua `torchrun`)
```bash
torchrun --nproc_per_node=2 train.py \
  --config configs/experiments/bhsd_mt_vnet_binary_fold0.yaml \
  --ddp
```

