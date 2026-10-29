# Quy ước checkpoint cho evaluation

Đối chiếu source ngày 29/10/2026. Chính sách này dành cho evaluation; [resume training](checkpoint-resume.md) dùng policy riêng. Audit gốc mô tả trạng thái trước bản sửa.

## Chọn checkpoint

`utils/checkpoint_resolver.py` là nơi dùng chung để tìm checkpoint:

1. Xét `fold_x` trước thư mục legacy `foldx`.
2. Trong fold: `self_train/`, root, rồi `pre_train/` nếu bật `--include_pretrain`.
3. Phase đầu tiên có checkpoint sở hữu fold; không lấy best của phase cũ thay cho latest của phase mới.
4. Trong phase, ưu tiên `best_model.pth`, `VNet_best_model.pth`, rồi `{model}_best_model.pth`.
5. Mặc định yêu cầu best. `--allow_latest` cho phép fallback latest kèm warning; `--checkpoint_preference latest` yêu cầu latest, không fallback ngược.

Không so `best_dice` để chọn giữa phase. Đây là điểm tốt nhất lịch sử, không phải điểm của trọng số hiện tại. Checkpoint legacy vẫn được nhận nhưng provenance được báo `unverified`.

Benchmark kiểm tra checkpoint của cả 5 fold trước inference. Mặc định thiếu fold sẽ raise và CLI trả exit code khác 0. `--on_missing skip` cho phép chạy một phần, ghi lý do trong `skipped_folds` và `complete=false`. Không có fold nào thì raise, không xuất kết quả điểm.

## Dựng và load model

Eval dùng `config.model` và `config.method` trong checkpoint; xử lý PICK + VNet thành `pick_vnet`. Checkpoint mới lưu `model_spec` từ builder dùng chung với training; checkpoint cũ khôi phục đúng tập tham số mà train cũ thực sự truyền. Override model mâu thuẫn với config gây lỗi. Checkpoint cũ thiếu config phải truyền `--model`; các tùy chỉnh kiến trúc không suy ra được từ checkpoint cũ chưa được hỗ trợ tự động.

Load bằng `weights_only=True`, trên CPU và `strict=True`, ưu tiên `ema_net` nếu có. Sai key hoặc shape đều gây lỗi. Không tự chuyển sang pickle không hạn chế nếu checkpoint không tương thích.

## Lưu và provenance

Flow chuẩn hiện ghi `best_model.pth`, không ghi thêm bản sao `VNet_best_model.pth`; tên legacy vẫn được resolver hỗ trợ đọc. Training chỉ tạo best khi validation cải thiện. Không còn tạo best từ trọng số cuối khi thiếu best. Checkpoint best mới có `checkpoint_kind=best`, `selection_metric=val_mDice`, `selected_iter`; latest mang `checkpoint_kind=latest`. Các file best legacy không được tự gán provenance đã xác minh.

Mỗi lần lưu checkpoint, ghi file tạm và replace, rồi cập nhật `checkpoint_manifest.json` trong cùng thư mục phase. Manifest chứa entry theo tên file, đường dẫn tương đối từ thư mục chứa manifest, kind, phase, iteration và metadata lựa chọn. Manifest hiện là dấu vết kiểm toán, chưa là nguồn discovery của resolver; không tự đổi tên checkpoint tùy ý thông qua manifest. Hai lần replace không phải một transaction chung; checkpoint là nguồn thực tế nếu tiến trình dừng giữa hai bước.

## Báo cáo

Mỗi fold ghi checkpoint, kind, provenance, model và nguồn trọng số (`net`/`ema_net`). Summary ghi fold đã chạy/bỏ qua và chính sách lựa chọn.

`fold_mean_dice` / `fold_std_dice` thay cho `overall_oof_mean_dice` / `overall_oof_std_dice`: trung bình không trọng số và độ lệch chuẩn qua fold, không phải thống kê gộp tất cả ca OOF. Consumer dùng schema cũ cần đổi tên trường.

## Ví dụ

PICK legacy, yêu cầu đủ 5 fold best:

```bash
python evaluate.py --model pick --dataset bhsd_multiclass \
  --patch_size 32 160 160 \
  --exp_dir /home/u001015/experiments/BHSD_multiclass_PICK_5folds
```

Trong inventory local được kiểm tra ngày 27/10, Mean Teacher chỉ có latest ở fold 0–1. Config trong hai checkpoint ghi `unet3d`, task `binary` (2 lớp), và chỉ 1–2 iteration; đây không phải run multiclass hoàn chỉnh. bỏ `--model` để tự dựng đúng kiến trúc. Chạy có chủ đích một phần:

```bash
python evaluate.py --dataset bhsd_binary \
  --patch_size 32 160 160 --checkpoint_preference latest --on_missing skip \
  --exp_dir /home/u001015/experiments/BHSD_mean_teacher_5folds
```

## Bằng chứng kiểm tra ngày 27/10/2026

- 17 test (9 regression test mới và 8 test batching) đã qua. Các test mới dùng checkpoint PyTorch thật, bao gồm chọn phase, legacy, fallback, manifest, strict load, config/EMA và summary thiếu fold.
- Resolver tìm đúng best PICK ở cả 5 fold; đã kiểm tra strict load cả 5 checkpoint thật và checkpoint Supervised fold 0. Mean Teacher fold 0–1 load strict thành công với UNet3D binary và trọng số EMA.
- Audit tiếp theo đã cô lập test framework vào thư mục tạm và chạy lại thành công. Xem [CRITICAL_LOGIC_AUDIT_2026_10_27.md](../archive/audits/critical-logic-2026-10-27.md) cho bộ test đầy đủ và các thay đổi training.
- Chưa chạy lại inference benchmark trên dataset; môi trường kiểm tra không có CUDA khả dụng.
