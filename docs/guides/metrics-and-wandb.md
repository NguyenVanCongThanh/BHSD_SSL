# Metrics và W&B

Đối chiếu source ngày 29/10/2026. Registry và semantics là nguồn tra cứu hiện hành; [thiết kế ban đầu](../archive/plans/wandb-design-before-doc-review-2026-10-29.md) được lưu riêng, không dùng roadmap cũ làm trạng thái triển khai.

## Định nghĩa và giới hạn metrics

Giữ nguyên công thức chọn best: trung bình foreground Dice qua các case. Khi prediction và GT cùng vắng class, Dice của class/case đó bằng `1`. Không đổi checkpoint selection ngầm khi thêm metric.

Bổ sung cho từng class:

| Trường | Ý nghĩa |
|---|---|
| `dice_case_mean` | Mean Dice trên toàn bộ case, gồm ca âm tính |
| `dice_positive_mean` | Mean Dice chỉ trên case có GT của class |
| `positive_cases`, `negative_cases` | Support để đọc đúng hai loại mean |
| `negative_fp_case_rate` | Tỷ lệ case GT âm tính có ít nhất một voxel FP |
| `negative_fp_voxels_mean` | Số voxel FP trung bình trên case âm tính |
| `negative_fp_ml_mean` | FP volume trung bình trên case âm tính, chỉ có khi đủ spacing |
| `precision_micro`, `recall_micro` | Precision/recall từ voxel counts gộp, không phải lesion detection sensitivity |
| `tp_voxels`, `fp_voxels`, `fn_voxels` | Counts native grid phục vụ kiểm toán |
| `hd95_mm_mean`, `asd_mm_mean` | Surface distances tùy chọn, chỉ mean trên cặp có cả hai mask nonempty |
| `surface_valid_pairs` | Số cặp đủ điều kiện tính surface distance |

Không có support hoặc denominator bằng 0 thì ghi `null`, không tự coi là score hoàn hảo hoặc lỗi 0. FP mL cần spacing Z/Y/X dương, đơn vị mm; voxel volume chia 1000 để thành mL.

`val_surface_metrics: true` yêu cầu spacing; thiếu spacing gây lỗi rõ ràng. Surface distance dùng bề mặt voxel với connectivity mặc định của SciPy, khoảng cách Euclidean theo spacing; HD95 là percentile 95 của hai tập khoảng cách gộp, ASD là mean đối xứng gộp. Cặp mask rỗng ghi `null` và trạng thái `both_empty`/`one_empty`, không bịa penalty 100 mm. Khi so sánh với thư viện khác phải thống nhất định nghĩa HD95/ASD và empty-mask policy.

Training validation, single evaluation và benchmark dùng chung hàm thống kê. Benchmark phân biệt unweighted mean qua fold với `oof_case_metrics` tính trên tất cả case đã đánh giá; nếu thiếu fold, kết quả vẫn là partial OOF, không được gọi là full 5-fold.

Ví dụ export single checkpoint:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pth \
  --dataset bhsd_multiclass --fold 0 \
  --metrics_output /tmp/evaluation_metrics.json
```

Thêm `--surface_metrics` khi dataset có spacing; `--no_amp` tắt evaluation autocast. Các setting TTA/patch/stride phải thống nhất khi so sánh điểm.

## 5. Metric Registry (Bảng Tra cứu Chỉ số Toàn diện)

Hệ thống hỗ trợ linh hoạt cả cấu hình **Binary** (1 lớp tổn thương) lẫn **Multiclass** (nhiều lớp tổn thương), danh sách tên lớp được nạp động từ cấu hình dataset metadata thay vì cố định 5 lớp.

| Metric Key | Nhóm / View | Đơn vị | Chiều tối ưu | Điều kiện tính hợp lệ | Summary Policy | Ý nghĩa Phân tích |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| `val/mDice` | Overview | 0..1 | Max $\uparrow$ | Toàn bộ ca val | `max` | Macro Mean Dice theo quy ước legacy |
| `val/macro_positive_dice` | Overview | 0..1 | Max $\uparrow$ | Ít nhất 1 lớp có $GT > 0$ | `max` | Macro Dice chỉ trên các ca thực sự có bệnh |
| `val/best_mDice` | Overview | 0..1 | Max $\uparrow$ | Theo phase hiện tại | `last` | Kỷ lục mDice của phase đang chạy |
| `val/dice_{class}` | Overview | 0..1 | Max $\uparrow$ | Toàn bộ ca val | `max` | Dice trung bình của từng lớp tổn thương |
| `val/{class}_dice_positive_mean` | Overview | 0..1 | Max $\uparrow$ | Lớp có $GT > 0$ | `max` | Positive Dice của từng lớp |
| `train/total` | Overview | scalar | Min $\downarrow$ | Mỗi training step | `min` | Tổng loss tối ưu |
| `train/supervised` | Overview | scalar | Min $\downarrow$ | Mỗi training step | `min` | Segmentation loss có giám sát |
| `train/consistency` | Overview | scalar | Min $\downarrow$ | Khi bật SSL | `min` | Semi-supervised consistency loss |
| `train/lr` | Overview | scalar | - | Mỗi training step | `last` | Learning rate tại thời điểm huấn luyện |
| `train/weight` | Overview | scalar | - | Khi bật ramp-up | `last` | Trọng số consistency loss |
| `val/{class}_hd95_mm_mean` | Evaluation | mm | Min $\downarrow$ | Cả Pred & GT không rỗng | `min` | 95th Hausdorff Distance |
| `val/{class}_asd_mm_mean` | Evaluation | mm | Min $\downarrow$ | Cả Pred & GT không rỗng | `min` | Average Surface Distance |
| `val/{class}_negative_fp_case_rate` | Evaluation | 0..1 | Min $\downarrow$ | Có ca $GT = 0$ | `min` | Tỷ lệ ca âm tính bị dự đoán nhầm |
| `val/{class}_negative_fp_ml_mean` | Evaluation | ml | Min $\downarrow$ | Có ca $GT = 0$ | `min` | Thể tích báo nhầm trung bình trên ca âm tính |
| `val/{class}_precision_micro` | Evaluation | 0..1 | Max $\uparrow$ | $TP + FP > 0$ | `max` | Voxel precision của lớp |
| `val/{class}_recall_micro` | Evaluation | 0..1 | Max $\uparrow$ | $TP + FN > 0$ | `max` | Voxel recall của lớp |
| `val/predictions` | Evaluation | media | - | Theo `val_visual_interval`| - | Ảnh composite trực quan (CT, GT, Pred, Error) |
| `val/evaluation_table` | Evaluation | table | - | Mỗi kỳ validation | `last` | Bảng tổng kết đa lớp kèm support counts |
| `val/{class}_tp_voxels` | Diagnostics | count | - | Không lọc | `last` | Tổng số voxel True Positive |
| `val/{class}_fp_voxels` | Diagnostics | count | - | Không lọc | `last` | Tổng số voxel False Positive |
| `val/{class}_fn_voxels` | Diagnostics | count | - | Không lọc | `last` | Tổng số voxel False Negative |
| `val/{class}_positive_cases` | Diagnostics | count | - | Không lọc | `last` | Số ca dương tính thực tế trong tập val |
| `val/{class}_negative_cases` | Diagnostics | count | - | Không lọc | `last` | Số ca âm tính thực tế trong tập val |
| `val/{class}_surface_valid_pairs` | Diagnostics | count | - | Cả Pred & GT không rỗng | `last` | Số cặp tính được khoảng cách bề mặt |

---

## 6. Chính sách Tương thích Ngược (Backward Compatibility)

Để bảo đảm tất cả các công cụ phân tích hiện hữu không bị gián đoạn, dự án thiết lập quy tắc tương thích nghiêm ngặt:

1. **Giữ nguyên toàn bộ Backend Logging:**
   - Tiếp tục ghi nhận các khóa: `val/mDice`, `val/best_mDice`, `val/dice_{class}` và `val/{class}_dice_case_mean`.
   - Các công cụ như [`tools/compare_runs.py`](../../tools/compare_runs.py), [`tools/fetch_latest_run.py`](../../tools/fetch_latest_run.py), [`tools/fetch_wandb.py`](../../tools/fetch_wandb.py), [`tools/analyze_run_9tvq8qld.py`](../../tools/analyze_run_9tvq8qld.py) tiếp tục hoạt động bình thường trên cả run cũ lẫn run mới.
2. **Không xóa dữ liệu trùng ở tầng log:**
   - Việc trùng lặp giữa `val/dice_{class}` và `val/{class}_dice_case_mean` được giải quyết ở tầng hiển thị: **chỉ đưa 1 đại diện vào Workspace View**, không tạo panel cho key còn lại.
   - Nếu thực hiện chuẩn hóa schema trong tương lai, phải tuân thủ quy trình 3 bước:
     * Bước 1: Thiết lập cơ chế alias/bảng ánh xạ trong reader.
     * Bước 2: Cập nhật reader hỗ trợ đọc song song cả 2 định dạng (fallback).
     * Bước 3: Thông báo lịch trình deprecation rõ ràng trước khi dừng ghi alias cũ.

---

## W&B: history, identity và lifecycle

### Trục thời gian

SDK history append-only. Manager log `global_step` như một metric và dùng `define_metric(..., step_metric='global_step')`; không ghi đè SDK history bằng training iteration. Train/val có thể là hai history row cùng `global_step`, không làm mất media vì cố ghi về một step cũ. `val/mDice` khai báo summary `max`; best/last được lưu riêng.

`tools/fetch_wandb.py` ghép các history row cùng `global_step` để xuất loss và validation tại đúng iteration. Run cũ dùng `iter`, rồi mới fallback `_step` khi thiếu cả hai trục; không lấy loss từ iteration khác. Các metric đếm per-class được đăng ký `hidden=True` bằng tên cụ thể ngay trước lần log đầu tiên, vì W&B không hỗ trợ wildcard ở giữa tên metric.

### Identity và resume

- Run ID được lưu trong `<fold_dir>/wandb_run.json` và checkpoint `wandb_run`.
- Lần chạy mới kiểm tra checkpoint trước khi tạo W&B run hoặc ghi identity; tất cả rank kiểm tra trước khi bọc model bằng DDP. PICK kiểm tra thư mục của phase được yêu cầu, nên vẫn có thể bắt đầu self-training mới từ checkpoint pre-training.
- Resume dùng lại ID này hoặc explicit `logging.wandb_id`; không tự tạo ID từ tên run rồi gọi đó là resume.
- Nếu file identity và checkpoint có ID/project/entity mâu thuẫn, resume dừng trước khi gọi W&B để tránh ghi vào sai run.
- Online resume dùng `resume='must'`. Mâu thuẫn ID/project/entity gây lỗi; không tự chuyển sang run mới khi run cũ bị xóa/xung đột.
- Offline giữ ID nhưng mỗi lần chạy có thể tạo một segment offline riêng. Việc hợp nhất lịch sử khi sync cần kiểm tra; không cam kết offline resume là append vào cùng file cũ.

### Lifecycle và lỗi

- Manager dùng run handle riêng, không gọi global logging/summary để tránh ghi nhầm run.
- Training exception đóng run với `exit_code=1`; run thành công đóng với code 0. Áp dụng cả BaseMethod và PICK.
- Init/log thất bại có warning. Log lỗi lặp được giới hạn tần suất; `wandb_required: true` yêu cầu fail thay vì tiếp tục local-only.
- Artifact online chờ xác nhận tối đa 60 giây trước khi dọn phiên bản cũ; log “queued” không phải bằng chứng upload đã hoàn thành. Xem [chính sách lưu trữ checkpoint](checkpoint-resume.md#local-và-cloud).

```yaml
logging:
  use_wandb: true
  wandb_mode: offline  # online | offline | disabled; WANDB_MODE env có ưu tiên
  wandb_required: true
```

Notebook/generator đã bỏ credential W&B literal; đọc `WANDB_API_KEY` từ environment hoặc Kaggle Secrets. Thiếu secret thì notebook báo rõ và chọn offline. Token đã có trong lịch sử Git trước đó **không bị xóa khỏi lịch sử bởi thay đổi này**; cần thu hồi/đổi token đó qua tài khoản sở hữu.

Config được redaction trước khi ghi W&B/local snapshot/checkpoint đối với các key credential thông dụng. Đây không phải bộ phát hiện mọi dạng bí mật; không đưa credential vào tên run, comment hoặc arbitrary string trong config.

### Tài liệu SDK

- [W&B Run.log, define_metric và finish](https://docs.wandb.ai/ref/python/experiments/run/)
- [Logging history và summary](https://docs.wandb.ai/models/support/difference_log_summary)


## Dashboard và bảng validation

Code hiện có bảng evaluation, macro positive Dice và cấu hình ba view `Overview`, `Evaluation`, `Diagnostics`. Bảng được dựng qua helper dùng chung; orchestration ở [validation coordinator](../../training/validation_coordinator.py), không cần bổ sung lại trực tiếp vào BaseMethod theo roadmap cũ.

[Script dashboard](../../tools/setup_wandb_dashboard.py) hỗ trợ xem trước cấu trúc bằng `--dry-run`; dùng `--help` để xem tham số theo phiên bản checkout. Mục tiêu là tái sử dụng/cập nhật view; cần xác minh hành vi trên project và phiên bản SDK thực tế trước khi khẳng định không có duplicate hay đã đặt view mặc định thành công.

`run.summary` không tự có cùng ý nghĩa cho mọi key: metric có thể dùng last, max hoặc trường selected được ghi tường minh. Không ghép extrema của các metric khác nhau thành kết quả một checkpoint. `val/mDice` dùng summary max theo manager; đánh giá checkpoint cần iteration và weights source tương ứng.

Bảng history và summary phục vụ cách xem khác nhau; khả năng tua bảng trên UI phụ thuộc SDK/UI và loại dữ liệu đã log, không phải cam kết từ một lệnh log. Code/test offline không chứng minh cloud rendering, upload hay đồng bộ segment đã đạt.

Xem [visualization](validation-visualization.md), [checkpoint evaluation](checkpoint-evaluation.md) và [run analysis](../README.md#phân-tích-run).
