# 📊 Phân Tích Cơ Chế Trực Quan Hóa (Visualization) Validation Predictions Trên W&B & Đề Xuất Cải Tiến

Tài liệu này phân tích chi tiết nguyên lý hoạt động, bản chất hiện tượng hiển thị, các hạn chế kỹ thuật hiện tại và giải pháp tối ưu hóa toàn diện cho module trực quan hóa kết quả kiểm định (**Validation Predictions**) của mô hình Mean Teacher trên nền tảng Weights & Biases (W&B).

---

## 1. Luồng Hoạt Động Của Module Visualize Hiện Tại

Cơ chế trích xuất và hiển thị kết quả kiểm định được phối hợp qua 3 module chính:

```
[ evaluation/sliding_window.py ]
   │  Chạy sliding-window inference 3D toàn bộ tập Validation
   │  Rank 0 gom 4 ca đầu tiên: {'case', 'image', 'label', 'pred', 'dice'}
   ▼
[ methods/base_method.py ]
   │  Gọi build_visual_samples(visual_records)
   │  Tạo class_labels = {0: 'background', 1: 'edh', 2: 'iph', 3: 'ivh', 4: 'sah', 5: 'sdh'}
   ▼
[ utils/visualizer.py ]
   │  1. extract_best_slice(): Tìm lát cắt Z có diện tích tổn thương lớn nhất theo Ground Truth
   │  2. standardize_2d_slice(): Resize lát cắt về chuẩn (512, 512)
   │  3. create_wandb_segmentation_samples(): Đóng gói thành wandb.Image kèm dual masks
   ▼
[ Weights & Biases Cloud ]
   └── Hiển thị tại mục val/predictions trên Dashboard
```

### Các bước xử lý chi tiết trong `utils/visualizer.py`:
1. **Chọn lát cắt Z tối ưu (`extract_best_slice`)**:
   - Đếm số lượng voxel tổn thương ($> 0$) trên từng lát cắt của khối 3D Ground Truth:
     $$\text{best\_z} = \arg\max_z \sum_{x, y} (\text{mask\_3d}[z, y, x] > 0)$$
   - Việc ưu tiên tìm lát cắt theo Ground Truth giúp chỉ số $Z$ giữ **cố định 100% qua tất cả các checkpoint** ($250, 500, 750, \dots, 30000$), cho phép quan sát cùng một lát cắt bệnh nhân cải thiện dần theo thời gian.
2. **Chuẩn hóa kích thước (`standardize_2d_slice`)**:
   - Sử dụng phép nội suy Bilinear (cho ảnh CT) và Nearest (cho nhãn mask) để đưa toàn bộ về kích thước $(512, 512)$, loại bỏ cảnh báo kích thước ảnh không đồng đều của W&B.
3. **Đóng gói đối tượng W&B Image**:
   ```python
   wandb.Image(
       slice_img,
       caption=caption,
       masks={
           "predictions":  {"mask_data": slice_pr, "class_labels": class_labels},
           "ground_truth": {"mask_data": slice_gt, "class_labels": class_labels}
       }
   )
   ```

---

## 2. Giải Mã Hiện Tượng: "Tại sao lúc mờ, lúc sáng, và thay đổi sau mỗi lần Val?"

Người dùng quan sát thấy: *Vùng tổn thương lúc thì màu rất mờ, lúc thì sáng bình thường, và sau mỗi lần validation lại biến đổi không đồng nhất.*

### Bản chất cốt lõi: W&B đang vẽ CHỒNG ĐÈ 2 lớp mặt nạ cùng lúc!
Panel `val/predictions` **không chỉ hiển thị riêng Ground Truth hay riêng Prediction**, mà đang nạp cả hai:
- Lớp 1: `ground_truth` (Vùng bác sĩ khoanh vùng chuẩn).
- Lớp 2: `predictions` (Vùng mô hình Teacher dự đoán tại checkpoint đó).

Mặc định trên giao diện W&B, cả 2 lớp này **được BẬT ĐỒNG THỜI** và máy tính thực hiện thuật toán **hòa trộn màu (Alpha Blending)**:

$$\text{Color}_{\text{display}} = \alpha \cdot \text{Color}_{\text{pred}} + (1 - \alpha) \cdot \text{Color}_{\text{gt}}$$

Từ cơ chế này, các hiện tượng cụ thể được giải thích như sau:

| Hiện tượng quan sát | Bản chất kỹ thuật bên dưới | Ý nghĩa Y tế & Machine Learning |
|---|---|---|
| **Vùng màu SÁNG RỰC, ĐẬM NÉT** | **True Positive (Trùng khớp)**: Cả `ground_truth` và `predictions` đều nhận diện đúng loại xuất huyết tại pixel đó. Hai lớp mask cùng màu đè khít lên nhau làm tăng gấp đôi độ đậm (Opacity). | Mô hình học tốt, dự đoán khớp hoàn toàn với chẩn đoán của bác sĩ. |
| **Vùng màu BỊ MỜ, NHẠT, XỈN MÀU** | **False Negative (Mô hình bỏ sót)**: Bác sĩ khoanh vùng có máu (ví dụ class 2 - IPH màu đỏ/tím), nhưng mô hình (ở iter 250 còn non) đoán vùng đó là **Background (não lành)**. Lớp mask Background màu xanh của Prediction bị phủ đè lên lớp màu của Ground Truth làm giảm tương phản, gây mờ đục. | Mô hình chưa học kịp tổn thương này, đang bị bỏ sót. |
| **Vùng màu LẠ XUẤT HIỆN Ở NÃO LÀNH** | **False Positive (Mô hình đoán thừa)**: Vị trí không có tổn thương trong Ground Truth nhưng mô hình dự đoán nhầm thành xuất huyết. | Mô hình bị dương tính giả (nhiễu). |
| **HÌNH ẢNH THAY ĐỔI SAU MỖI LẦN VAL** | Nhãn Ground Truth là **bất biến**, nhưng trọng số của mô hình **liên tục được cập nhật**. Sau mỗi 250 bước, `pred_3d` thay đổi $\rightarrow$ mức độ giao thoa giữa `pred` và `gt` thay đổi theo. | Thể hiện quá trình mô hình đang học và tiến hóa dần qua từng bước huấn luyện. |

---

## 3. Các Điểm Chưa Tối Ưu (Lỗ Hổng Logic) Trong Code Hiện Tại

### Hạn chế 1: Đưa nhãn `background` (Class 0) vào danh sách mask
Trong file `methods/base_method.py`:
```python
class_labels = {i: name for i, name in enumerate(self.class_names)}
# Kết quả: {0: 'background', 1: 'edh', 2: 'iph', 3: 'ivh', 4: 'sah', 5: 'sdh'}
```
- **Hậu quả**: Khi `0: 'background'` xuất hiện trong từ điển nhãn, W&B xem giá trị 0 là một nhóm đối tượng cần tô màu (thay vì coi là nền trong suốt). W&B tự động phủ một lớp màng màu xanh dương lên 99% diện tích (não lành và không khí).
- Lớp màu xanh mờ này làm tối ảnh CT, làm giảm độ tương phản của xương sọ và hòa lẫn vào các tổn thương thật.

### Hạn chế 2: Chồng đè 2 mặt nạ lên cùng 1 tấm ảnh (Single-Canvas Overlap)
- Khi cả hai mask cùng bật trên một canvas, mắt người không thể bóc tách độc lập đâu là ranh giới do bác sĩ vẽ, đâu là ranh giới do AI dự đoán. Người xem dễ lầm tưởng toàn bộ bức tranh là Ground Truth bị lỗi hiển thị.

### Hạn chế 3: Chuẩn hóa Min-Max đơn thuần thay vì CT Brain Windowing
Trong `utils/visualizer.py`:
```python
min_val, max_val = slice_img.min(), slice_img.max()
slice_img = (slice_img - min_val) / (max_val - min_val)
```
- **Hậu quả**: Xương sọ có độ hấp thụ tia X cực lớn ($+1000$ đến $+2000$ Hounsfield Units - HU), trong khi dải tổn thương xuất huyết não chỉ nằm ở khoảng $+50$ đến $+90$ HU, não lành $+20$ đến $+40$ HU.
- Phép chia cho `max_val` bị chi phối bởi xương sọ, nén toàn bộ mô não và máu tụ vào một dải xám hẹp, làm giảm độ sắc nét của bệnh lý bên dưới.

---

## 4. Kế Hoạch & Đề Xuất Cải Thiện Toàn Diện

### Đề xuất 1: Thao tác người dùng trên giao diện W&B (Ngay lập tức)
Nếu không muốn sửa code ngay, bạn có thể kiểm soát hiển thị trên Dashboard W&B:
1. Nhấp chuột vào ảnh để mở to.
2. Tại thanh công cụ **Masks** ở bên phải:
   - **Bỏ tích chọn `background`**: Lớp màng xanh dương sẽ biến mất, trả lại nền CT nguyên bản đen trắng sắc nét.
   - **Bật riêng `ground_truth`**: Xem chính xác 100% vùng tổn thương thực tế cố định của bệnh nhân.
   - **Bật riêng `predictions`**: Xem độc lập kết quả mô hình AI khoanh vùng tại vòng lặp đó.

---

### Đề xuất 2: Cải tiến Code – Chuẩn hóa Class Labels (Loại bỏ Background khỏi Mask)
Trong `methods/base_method.py`, chỉ truyền các nhãn từ Class 1 trở đi vào W&B:
```python
# Chỉ giữ các lớp bệnh lý (1..5), loại bỏ class 0 (background)
class_labels = {i: name for i, name in enumerate(self.class_names) if i > 0}
```
👉 **Tác dụng**: W&B sẽ tự động coi toàn bộ pixel có giá trị 0 là **HOÀN TOÀN TRONG SUỐT (Transparent 100%)**. Chỉ những nơi có xuất huyết mới sáng màu rực rỡ, ảnh CT nền giữ nguyên độ trung thực.

---

### Đề xuất 3: Tạo Khung Hình So Sánh Song Song 3 Bảng (Tri-Panel Comparison)
Thay vì bắt người xem phải bật/tắt thủ công trên W&B, nâng cấp `utils/visualizer.py` để sinh ra một ảnh tổng hợp gồm 3 khung hình đặt cạnh nhau (Side-by-Side):

$$\boxed{\text{Ảnh CT Não Gốc}} \quad \Big| \quad \boxed{\text{Ground Truth (Bác sĩ)}} \quad \Big| \quad \boxed{\text{Dự đoán AI (Teacher Model)}}$$

#### Bảng màu chuẩn y tế cố định (Color Palette):
- Class 1 (EDH - Ngoài màng cứng): Đỏ rực `[230, 25, 75]`
- Class 2 (IPH - Nội nhu mô): Xanh lá `[60, 180, 75]`
- Class 3 (IVH - Trong não thất): Vàng `[255, 225, 25]`
- Class 4 (SAH - Dưới nhện): Tím `[145, 30, 180]`
- Class 5 (SDH - Dưới màng cứng): Cam `[245, 130, 48]`

#### Lợi ích vượt trội:
1. **Minh bạch 100%**: Nhìn thấy ngay lập tức mô hình dự đoán đúng chỗ nào, sai chỗ nào mà không cần click chuột.
2. **Chuẩn mực xuất bản**: Định dạng 3 panel ghép ngang là quy chuẩn bắt buộc của các hội nghị, tạp chí y tế hàng đầu (MICCAI, IEEE TMI, Nature Medicine).
3. **Giữ nguyên tính năng tương tác**: Vẫn có thể giữ layer W&B Interactive Mask song song với ảnh Tri-Panel tĩnh.

---

### Đề xuất 4: Tích hợp Brain Windowing (+40 HU, Width 80 HU)
Áp dụng cửa sổ hiển thị mô não chuẩn trước khi vẽ:
```python
def apply_brain_window(slice_img, window_center=40, window_width=80):
    min_hu = window_center - window_width / 2  # 0 HU
    max_hu = window_center + window_width / 2  # 80 HU
    windowed = np.clip(slice_img, min_hu, max_hu)
    return (windowed - min_hu) / (max_hu - min_hu)
```
👉 Vùng máu tụ (50 - 80 HU) sẽ sáng trắng nổi bật tự nhiên trên nền nhu mô não xám, giúp bác sĩ và chuyên gia đánh giá trực quan cực kỳ chuẩn xác.

---

## 5. Tổng Kết

| Đặc điểm | Hệ Thống Hiện Tại | Sau Khi Nâng Cấp Theo Đề Xuất |
|---|---|---|
| **Hiển thị Background** | Bị tô màng xanh mờ toàn bộ ảnh CT | Nền trong suốt 100%, ảnh CT đen trắng rõ nét |
| **Cách so sánh** | Chồng đè 2 mask gây mờ và lẫn lộn màu | Hiển thị 3 khung hình song song (Gốc \| Bác sĩ \| AI) |
| **Độ ổn định trực quan** | Cảm giác mờ/sáng thất thường sau mỗi lần Val | Thấy rõ ranh giới thật cố định và AI tiến bộ từng bước |
| **Độ tương phản ảnh CT** | Min-Max toàn ảnh bị kéo dãn bởi xương | Brain Windowing làm nổi bật tự nhiên khối máu tụ |
