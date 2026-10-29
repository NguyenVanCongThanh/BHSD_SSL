# Giao thức benchmark

Đối chiếu pipeline trong repo ngày 29/10/2026. Hỗ trợ một tùy chọn inference không chứng minh toàn pipeline đã tương đương nnU-Net. Các nhận xét thiết kế cũ nằm trong [archive](../archive/plans/benchmark-alignment-before-doc-review-2026-10-29.md).

## Điều kiện cần ghi nhận

| Hạng mục | Cần thống nhất và báo cáo |
|---|---|
| Dữ liệu | Patient IDs, fold, labeled set, unlabeled pool và preprocessing version |
| Không gian metric | Grid, spacing, nhãn, empty-mask policy và cách tổng hợp case/fold |
| Inference | Patch, stride, Gaussian, TTA, AMP và nguồn trọng số EMA/student |
| Chọn checkpoint | Metric chọn best, iteration, phase và provenance |
| Ngân sách | Optimizer updates, labeled exposure, batch, GPU time và seed |

Pipeline hiện đánh giá prediction/GT trên grid dữ liệu H5 được nạp. Xuất mask và đánh giá trên không gian NIfTI gốc chưa được xác nhận trong pipeline hiện hành; không đánh dấu hoàn tất hạng mục này. So sánh preprocessing cần đọc [dataset preprocessing](../architecture/dataset-preprocessing.md), không suy tương đương toàn bộ từ một ca mẫu.

## Chạy evaluation

Gaussian mặc định bật; TTA mặc định tắt. Thêm `--use_tta` khi protocol đã chọn TTA cho các bên; `--no_gaussian` dành cho ablation có ghi nhận.

```bash
python evaluate.py --dataset bhsd_multiclass \
  --root_path /path/to/BHSD_fold0 \
  --checkpoint /path/to/best_model.pth \
  --patch_size 32 192 192 --use_tta \
  --metrics_output /tmp/evaluation_metrics.json

python evaluate.py --dataset bhsd_multiclass \
  --root_path /path/to/BHSD_fold0 \
  --exp_dir /path/to/experiment --patch_size 32 192 192 --use_tta
```

Model được dựng từ checkpoint config/model spec khi có; không ép `--model nnunet3d` cho checkpoint kiến trúc khác. Xem [checkpoint evaluation](checkpoint-evaluation.md) cho best/latest, legacy và missing folds. Thêm `--surface_metrics` chỉ khi dataset cung cấp spacing hợp lệ.

## Thiết kế so sánh

So SSL với supervised nội bộ dưới backbone, nhãn, augmentation và ngân sách được kiểm soát; ghi rõ khác biệt do phương pháp. Có thể báo cáo thêm baseline nnU-Net theo recipe riêng cùng chi phí thực tế. Không gọi hai recipe là matched-budget nếu số updates, exposure hoặc compute khác nhau.

Báo cáo Dice case cùng positive Dice, support, precision/recall và FP ca âm; surface metrics cần cùng định nghĩa/spacing. Trung bình qua fold không thay thế thống kê gộp theo ca. Kiểm định thống kê cần prediction/metric ghép cặp thực tế và quy trình định trước; chưa có bằng chứng chạy thì không đánh dấu hoàn tất.

Không có bảo đảm SSL hội tụ sau 20–30k bước, tăng batch sẽ tăng throughput theo tỷ lệ cố định, hoặc recipe nào chắc chắn thắng baseline. Cần ablation và nhiều seed/fold phù hợp nguồn lực. Xem [metrics](metrics-and-wandb.md) và [tối ưu training](training.md).
