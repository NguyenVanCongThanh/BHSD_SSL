# Chẩn đoán lỗi Kaggle và DDP

Đối chiếu code ngày 29/10/2026. Phân biệt notebook chưa tới launcher, lỗi khởi tạo process group và treo khi đang train; log trống không tự chứng minh NCCL deadlock.

## Thu thập bằng chứng

1. Lấy log Kaggle cùng trạng thái phiên, không chỉ `output.log` W&B. Launcher có thể lỗi sau khi W&B ngừng ghi.
2. Ghi source revision/resolved config, số rank, iteration cuối và thời điểm checkpoint local/cloud.
3. Khi treo, đối chiếu tiến độ collective và log theo rank; xem CPU/RAM theo cây tiến trình, I/O và dung lượng working thực tế.
4. Phân biệt thông báo NCCL timeout, CUDA OOM, ENOSPC, cancel và lỗi subprocess. Không suy nguyên nhân từ một chỉ số memory/utilization.

## Khởi tạo và log

[core/distributed.py](../../core/distributed.py) hiện đặt process-group timeout 1800 giây và in log trước/sau barrier khởi tạo. Tăng timeout không sửa được nguyên nhân rank ngừng tiến triển.

Notebook/generator cần stream stdout, flush log quan trọng, dùng chế độ Python unbuffered và tránh Git chờ prompt ngầm. Kiểm tra log cài dependency và clone trước khi kết luận train bị treo. Cổng rendezvous không xung đột và NCCL settings phải được kiểm tra theo notebook/runtime đang dùng; không áp dụng một tổ hợp biến môi trường như bảo đảm cho mọi máy.

## Validation và bộ nhớ

Validation hiện chạy phân tán qua [runner](../../training/validation_runner.py) và [coordinator](../../training/validation_coordinator.py), không phải mọi validation chỉ do rank 0 chạy trong khi rank khác luôn đứng chờ. Các rank phải tham gia collective theo cùng lịch, kể cả rank không có case hoặc không log W&B.

Kiểm tra [global/per-rank batch](training.md), worker/prefetch, volume inference và bộ nhớ đỉnh. `gc.collect()` không giải phóng object còn tham chiếu; `torch.cuda.empty_cache()` không giải phóng tensor CUDA đang được dùng. Không coi thêm hai lệnh này là bản sửa memory leak khi chưa xác định nguồn giữ bộ nhớ.

## Sự cố có bằng chứng

[Run s9i9nsui ngày 23/10](../analysis/run-s9i9nsui-termination-2026-10-26.md) kết thúc với NCCL timeout; nguyên nhân rank ngừng tiến triển chưa xác định đầy đủ. Báo cáo cũ quy đầy đĩa/SIGKILL là nguyên nhân chính đã bị thay thế. Batch của run lịch sử cũng khác quy ước global batch sau sửa; không dùng nó làm trạng thái code mới.

Khi tái hiện, thêm heartbeat trước/sau lấy batch, forward, backward, optimizer, validation và save; thu trace theo thông báo runtime. Chỉ thay đổi tham số sau khi có bằng chứng và lưu lại cấu hình đã chạy.

Xem [submit Kaggle](kaggle.md), [checkpoint/resume](checkpoint-resume.md), [crawl dữ liệu](experiment-data.md).
