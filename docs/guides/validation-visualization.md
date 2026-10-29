# Hiển thị validation và cohort cố định

Cập nhật: 27/10/2026. Tài liệu này mô tả hành vi đã triển khai, thay cho các giả định về màu sắc của bản phân tích ban đầu.

## 1. Những kết luận cần đính chính

- Không thể suy ra TP/FN từ việc vùng overlay sáng hay mờ. Màu hiển thị còn phụ thuộc opacity, thứ tự layer, palette, background và lỗi nhầm class. Muốn xác định sai sót phải so nhãn prediction với GT.
- Không có bảo đảm rằng bỏ class `0` khỏi `class_labels` khiến pixel `0` trong suốt. Mapping tên lớp không phải hợp đồng điều khiển alpha của UI.
- Hai lớp cùng màu chồng nhau không đơn giản làm opacity tăng gấp đôi. Nếu hai lớp có cùng alpha `a`, opacity tổng trong phép compositing thông thường là `1-(1-a)^2`.
- Palette là quy ước cố định của dự án; không tuyên bố đó là bảng màu y tế bắt buộc. Layout nhiều panel là lựa chọn giúp so sánh, không phải quy định chung của tạp chí/hội nghị.
- Bản cũ chọn lát cắt theo prediction khi GT rỗng, nên không cố định Z qua checkpoint. Bản mới bỏ fallback đó.

## 2. Intensity: không áp dụng HU window lên tensor normalized

Ba file H5 validation đã kiểm tra có intensity xấp xỉ `-4..4`, không phải HU trực tiếp. Một file lưu:

```text
norm_params = {'mean': 47.6804, 'std': 18.4999}
clip_range = [11.0, 88.0]
target_spacing = (5.0, 0.5, 0.5)
```

Metadata normalization không tự chứng minh toàn bộ pipeline có thể đảo chính xác: thứ tự clipping, interpolation và padding có thể làm mất hoặc đổi giá trị. Không tự suy đơn vị từ numeric range hoặc riêng `mean/std`.

**Mặc định an toàn:** `intensity_mode: percentile`, dùng percentile của toàn volume để giữ display range cố định cho cùng case. Không dùng range của prediction hoặc min/max riêng từng lần chọn lát cắt.

**HU mode:** chỉ cho phép khi đã khai báo một trong hai điều kiện:

- `data.intensity_units: HU`: array là HU trực tiếp.
- `data.normalization: zscore_hu`: người chuẩn bị dữ liệu đã xác nhận phép biến đổi là z-score HU; H5 phải có `norm_params.mean/std` hợp lệ. Phép đảo là `image * std + mean`.

Không tự khai báo `zscore_hu` cho dataset hiện có nếu chưa xác minh preprocessing. Dữ liệu đã clip không thể phục hồi giá trị bị cắt. Window center/width là setting hiển thị cấu hình được, không phải thay đổi input training hay hướng dẫn chẩn đoán.

Dataset đưa `spacing`, `norm_params`, `clip_range` và khai báo units/normalization vào sample validation/test. Không thêm metadata này vào sample training để tránh lỗi default collation của DataLoader.

## 3. Hình ảnh so sánh

`utils/visualizer.py` tạo bốn panel cùng một canvas:

```text
CT | GT | Prediction | Error map
```

- CT/GT/Prediction dùng cùng display range và cùng phép biến đổi không gian.
- Overlay chỉ tác động foreground; background trong ảnh RGB giữ nguyên giá trị CT.
- Resize giữ tỷ lệ vật lý Y/X khi có spacing, rồi padding về canvas chung. Không kéo ảnh chữ nhật thành vuông.
- Nhãn resize bằng nearest; ảnh dùng bilinear.
- Error map: TP đúng class = xanh lá; FN = đỏ; FP = vàng; foreground bị nhầm class = tím hồng. Đây là các nhóm lỗi loại trừ nhau trong ảnh, không phải một bảng đếm one-vs-rest.
- Caption phân biệt **volume Dice 3D** và **slice Dice 2D**, ghi case ID, Z và display mode. Dice vẫn theo policy tại mục 5.

`val/predictions` chứa ảnh tổng hợp RGB. Nếu bật interactive, ảnh mask vuông được log riêng ở `val/predictions_interactive`; không trộn hai kích thước vào một gallery. UI interactive vẫn có thể cho bật/tắt lớp và background; ảnh RGB là view có rendering xác định.

Các ảnh hiển thị không được dùng để tính metric: metric chạy trên prediction và GT ở native evaluation grid, trước resize/padding hiển thị.

## 4. Cohort cố định, ổn định khi đổi số GPU

- Có thể khai báo `logging.visualization.case_ids` để chọn một nhóm case cố định, gồm lớp hiếm, tổn thương nhỏ và case âm tính.
- Nếu không khai báo, chọn các case ID đầu tiên theo thứ tự sắp xếp. Đây là fallback xác định, **không bảo đảm đại diện đủ các lớp**.
- Với GT có foreground, chọn Z có tổng foreground lớn nhất. Với GT rỗng, dùng lát giữa volume; prediction không được quyết định Z.
- `slice_indices` cho phép cố định Z theo case ID, kể cả lát dành cho lớp hiếm vốn không có diện tích lớn nhất.
- Mỗi rank suy luận phần case của mình và render thumbnail 2D cho case được chọn. Chỉ thumbnail được gather về rank 0; không gather toàn bộ ảnh/prediction 3D.
- Rank 0 công bố số media cần thu qua collective để các rank tham gia cùng một lịch, kể cả rank không log W&B.

Cohort nằm trong `validation_latest.json`. Khi cần chọn cohort có phân tầng, hãy xác lập danh sách một lần trước khi so sánh run. Chưa triển khai tự động chọn worst cases, montage mọi lát cắt hoặc lesion matching; không dùng ảnh đẹp được chọn lại theo checkpoint làm bằng chứng tổng quát.

## 5. Metric trên ảnh

Caption phân biệt volume Dice 3D và slice Dice 2D. Xem [định nghĩa metrics](metrics-and-wandb.md) cho empty masks, support và surface distances; [benchmark](benchmark-protocol.md) cho protocol evaluation. Render/resize không tham gia tính metric.

## 6. Config thực sự có tác dụng

```yaml
logging:
  val_amp: true
  val_gaussian: true
  val_tta: false
  val_surface_metrics: false
  val_visualize: true
  num_vis_samples: 4
  visual_every_n_validations: 1
  log_interval: 10
  artifact_interval: null
  visualization:
    case_ids: []
    slice_indices: {}
    target_shape: [512, 512]
    intensity_mode: percentile
    percentiles: [1.0, 99.0]
    window_center: 40.0
    window_width: 80.0
    alpha: 0.5
    interactive: false
```

- `val_interval`: lịch inference/metric; `visual_every_n_validations`: lịch media trên các lần validation định kỳ. Validation cuối phase ngoài lịch vẫn tính metric, không bắt buộc upload ảnh.
- `save_interval`: local latest; `artifact_interval`: xét upload latest tại các local save/validation, cộng cuối phase. Khi `null`, dùng `val_interval`. Best upload khi được chọn. Muốn upload đúng một chu kỳ cụ thể, chọn interval chia hết cho lịch save hoặc validation.
- Training scalar log theo `log_interval`; TensorBoard dùng cùng global step với W&B.
- Mean Teacher dùng chung evaluation options với BaseMethod, chỉ khác model được đánh giá là EMA teacher.
- Deep-supervision weights được dùng thực sự: list bao gồm **main + tất cả auxiliary heads**, normalize theo tổng. Sai length, giá trị âm/không hữu hạn hoặc tổng 0 gây lỗi. nnUNet 7 stages có main + 5 auxiliary heads, nên config mặc định đã sửa thành 6 weights.

`resolved_config.json` lưu config sau merge cùng batch thực tế mỗi rank, model spec khi entrypoint cung cấp, số labeled case thực tế, split hashes, inference options, source revision, trạng thái workspace dirty và phiên bản Python/PyTorch. Git revision riêng không chứng minh workspace sạch; cần commit code dùng cho run hoặc lưu diff khi công bố. Split hash không chứng minh patient-level independence hoặc hash toàn bộ dataset.

Metadata preprocessing và fingerprint khai báo trong config vẫn cần người tạo dataset xác nhận. Không tự chuyển UI display settings thành model preprocessing.

## 7. Logging và resume

Xem [W&B và metric logging](metrics-and-wandb.md) cho global step, run identity, lifecycle và lỗi; [checkpoint/resume](checkpoint-resume.md) cho nguồn khôi phục local/cloud.

## 8. Bằng chứng kiểm chứng ngày 27/10/2026 và giới hạn

Kết quả kiểm tra đợt này: **71 test qua**, gồm test W&B SDK **0.31.0** offline thật và DDP CPU/Gloo. Kiểm tra cú pháp notebook, compileall và `git diff --check` cũng qua.

Regression tests trong `tests/test_validation_tracking.py` kiểm tra intensity units, negative-case Z, aspect ratio, RGB background, wrong-class errors, positive/negative metric semantics, spacing, DS weights, run identity, global axis, failure status, redaction và metadata H5.

Có test SDK thật chỉ bật khi yêu cầu:

```bash
BHSD_TEST_WANDB_OFFLINE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -q
```

Test SDK dùng offline, synthetic images và thư mục tạm; không cần key, không login hoặc gửi dữ liệu lên cloud. DDP hai process còn kiểm tra cohort chọn case do rank 1 sở hữu được gather đúng về rank 0. Môi trường sandbox cần quyền IPC/loopback để chạy hai kiểm thử này.

Chưa xác minh giao diện interactive trên cloud, upload online, sync offline segments, CUDA AMP/NCCL, hiệu quả hội tụ hoặc khả năng hoàn thành một Kaggle run dài. Phần sửa notebook trong đợt này chỉ xử lý W&B authentication; những blocker PICK batch/resume/eval đã nêu ở lượt kiểm tra Kaggle trước đó không được tự coi là đã xử lý.
