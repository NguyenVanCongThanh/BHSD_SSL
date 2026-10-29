# Checkpoint và tiếp tục huấn luyện

Đối chiếu source ngày 29/10/2026: [BaseMethod](../../methods/base_method.py), [serializer](../../utils/checkpoint.py), [metadata](../../training/checkpoint_metadata.py), [W&B manager](../../utils/wandb_manager.py). [Audit gốc](../archive/audits/checkpoint-resume-2026-10-29.md) là đặc tả trước triển khai, không phải trạng thái hiện hành.

## Chọn checkpoint phục hồi

| File | Vai trò |
|---|---|
| `checkpoint_latest.pth` | Trạng thái theo lịch save/validation và cuối phase |
| `checkpoint_epoch_latest.pth` | Trạng thái sau batch cuối của một lượt DataLoader hoàn chỉnh |
| `best_model.pth` | Trọng số đạt validation best; dùng cho evaluation |

```bash
python train.py --dataset bhsd_multiclass --method mean_teacher --fold 0 \
  --data.root_path /path/to/BHSD_fold0 \
  --experiment.exp_dir /path/to/experiments --experiment.name mt_fold0 \
  --resume --experiment.resume_policy epoch_boundary
```

Giữ cùng tên run, output, split và cấu hình gốc. `experiment.resume_policy` nhận `latest` (mặc định) hoặc `epoch_boundary`; `--resume` phải được bật. Policy epoch yêu cầu file epoch riêng; thiếu file sẽ lỗi sau bước thử cloud restore, không âm thầm dùng latest. Không bật resume khi muốn run mới; dùng tên/thư mục mới để tránh checkpoint guard.

Checkpoint nằm tại `<exp_dir>/<experiment.name>/fold_<fold>/`; PICK có thư mục phase riêng `pre_train/`, `self_train/`. Đường dẫn Kaggle tùy CLI notebook, thường dưới `/kaggle/working/experiments`.

## Hành vi đã có trong code

Loop lưu epoch checkpoint sau batch cuối theo `len(trainloader)`, kèm `completed_epochs`, `epoch_length`, `at_epoch_boundary`, schema version và metadata batch/split khi có. Resume đọc `completed_epochs`; nếu thiếu thì suy bằng `iter_num // epoch_len`. Sampler DDP nhận epoch được khôi phục.

Giới hạn tổng iteration vẫn được giữ. Nếu đích 30000, epoch dài 77 và không có lịch sử resume làm lệch ranh giới, boundary cuối là 29953; bước 30000 không tự trở thành boundary. Latest giữa epoch tạo iterator mới khi resume, không tiếp tục cursor cũ.

Flow chuẩn hiện ghi best canonical, không ghi thêm bản sao VNet; resolver và PICK vẫn đọc tên legacy. Metadata latest/best đi qua các nhánh callback khác nhau: không giả định mọi checkpoint đều có đầy đủ epoch metadata chỉ vì epoch checkpoint có chúng.

## Local và cloud

Chỉ main process ghi checkpoint. File được ghi tạm rồi replace; manifest cập nhật sau, không phải một transaction chung. Local là nguồn đầu tiên khi resume; code không so iteration local với cloud để chọn bản mới hơn.

Upload cần W&B bật, `logging.save_wandb_model=true` và run hoạt động:

- Chỉ upload loại resume được chọn bởi `experiment.resume_policy`; cả hai file local vẫn được lưu.
- Latest dùng artifact `<prefix>_latest`; lịch xét qua `artifact_interval` (mặc định 5000) tại các lần save/validation, cộng cuối phase.
- Epoch dùng `<prefix>_epoch_latest`; `logging.epoch_artifact_interval` tính theo số epoch hoàn thành, mặc định mỗi 10 epoch. Phase kết thúc giữa epoch không tạo/upload boundary giả.
- Best dùng `<prefix>_best` khi validation cải thiện.

Online: sau khi queue upload, manager chờ xác nhận tối đa 60 giây rồi mới dọn phiên bản cũ. Timeout/lỗi xác nhận giữ nguyên phiên bản cloud; offline chỉ queue, không chờ hoặc dọn. Vì vậy log “queued” không chứng minh artifact đã tồn tại đầy đủ trên cloud. Kill/cancel không bảo đảm lưu khẩn cấp.

`logging.artifact_keep_versions: 2` giữ hai phiên bản mới nhất của run hiện tại trong mỗi collection (kể cả best); `0` tắt pruning. Không xóa phiên bản của run khác hoặc bản còn alias `latest`/`best`. Alias lịch sử `epoch_N` không ngăn pruning. Chỉ dọn collection vừa upload thành công; collection cũ không còn được upload sẽ không tự được dọn. Cleanup lỗi chỉ cảnh báo, không dừng train. Best vẫn chứa đầy đủ trạng thái để giữ tương thích với luồng resume hiện tại.

Cache artifact local được dọn bằng CLI W&B về `logging.artifact_cache_size: "1GB"` mỗi `artifact_cache_cleanup_interval: 5` upload đã xác nhận và sau khi finish online. Đặt kích thước `null` để tắt. Không dọn thư mục checkpoint, log run hoặc file staging/temp upload. Đây là mục tiêu kích thước cache, không phải giới hạn tổng dung lượng ổ đĩa.

## Giới hạn còn lại

Không có sampler cursor/permutation, worker RNG/prefetch hay RNG đầy đủ từng rank để bảo đảm exact resume. Metadata split/batch không đồng nghĩa đã có kiểm tra tương thích đầy đủ khi đổi split, world size hoặc batch. Fallback suy epoch của checkpoint legacy không chứng minh checkpoint đó thực sự ở boundary.

Không tự sửa iteration để biến checkpoint giữa epoch thành boundary. Khi quay lại checkpoint cũ, lịch sử W&B và best trên đĩa có thể đã đi xa hơn; cần kiểm tra nguồn checkpoint và run identity trước khi đánh giá.

Có [test epoch checkpoint](../../tests/test_epoch_boundary_checkpoint.py), nhưng tài liệu này không tuyên bố đã nghiệm thu GPU/NCCL, upload cloud hoặc mọi tổ hợp resume. Xem [checkpoint evaluation](checkpoint-evaluation.md) cho quy tắc chọn model đánh giá.
