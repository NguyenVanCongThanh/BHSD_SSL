# Kiểm tra trực tiếp W&B và Kaggle ngày 26/10/2026

## Kết luận đã xác nhận

- W&B vẫn ghi run mới nhất là `s9i9nsui`, `train/iter=3996`, state `finished`.
- Kaggle notebook `thnhnguynvncng/kaggle-bhsd-mean-teacher-dual-gpu` có trạng thái `ERROR`.
- Log Kaggle chứa lỗi NCCL ALLREDUCE timeout sau 1.800.073 ms (30 phút), sequence 28026, rank 1, NumelIn=629060.
- Rank 1 kết thúc bằng SIGABRT (exitcode -6). Torchrun gửi SIGTERM cho rank 0 (exitcode -15), sau đó trả exit code 1. Notebook phát sinh PapermillExecutionError.
- Log W&B không chứa đoạn lỗi launcher này, giải thích vì sao phân tích chỉ dựa trên output.log W&B đã bỏ sót nguyên nhân trực tiếp.
- Rank 1: last enqueued=28032, last completed=28025. Rank 0: last enqueued=28025, last completed=28024. Hai rank đã lệch tiến độ collective.
- Log console cuối tại iter 3990, Kaggle elapsed 13493.20 giây. Timeout được báo tại 15310.88 giây, cách khoảng 30 phút 18 giây. Phù hợp treo ở vùng iter 3996 rồi chờ watchdog.

## Số liệu tài nguyên

Từ 2040 bản ghi events W&B (1020 mẫu CPU/RAM và 1020 mẫu GPU):

| Chỉ số | Đầu run | Cực đại | Cuối run |
|---|---:|---:|---:|
| RAM hệ thống (%) | 22.87 | 96.17 | 94.90 |
| RSS tiến trình W&B theo dõi (MiB) | 2417.72 | 3614.96 | 2546.22 |
| RAM khả dụng (MiB) | 23695.53 | — | 1567.16 |
| GPU 0 memoryAllocatedBytes (GiB) | 10.33 | 10.73 | 10.73 |
| GPU 1 memoryAllocatedBytes (GiB) | 10.33 | 10.73 | 10.73 |

Cuối giai đoạn treo, GPU 0 utilization=0%, GPU 1=100%. Không thể diễn giải GPU 1=100% là còn train tiến triển: dữ liệu này cũng phù hợp chờ giao tiếp NCCL.

RAM hệ thống cao là yếu tố rủi ro có bằng chứng. Tuy nhiên RSS tiến trình được theo dõi gần như ổn định khoảng 2.5 GiB; chưa xác định tiến trình khác, worker, cache hay bộ nhớ kernel đóng góp bao nhiêu. Không đủ bằng chứng kết luận Python giữ lại toàn bộ volume validation hoặc OOM killer đã giết tiến trình.

Chỉ số disk W&B là `system.disk./.usageGB` khoảng 7108–7120 GiB, usage khoảng 87.1–87.3%. Đây không phải phép đo riêng quota `/kaggle/working`, nên không xác nhận cũng không loại trừ đầy quota working. Không có thông báo ENOSPC trong log đã lấy.

## Đối chiếu mã thực sự chạy

Đã tải 5 file từ output Kaggle vào `experiment_data/kaggle/thnhnguynvncng/kaggle-bhsd-mean-teacher-dual-gpu/<snapshot>/source/BHSD_SSL/`. File `methods/base_method.py` và `data/samplers.py` giống bản workspace khi đối chiếu bằng diff.

DDP truyền nguyên batch_size=4 và labeled_bs=2 vào sampler mỗi rank. Do đó mỗi GPU xử lý 4 mẫu (2 labeled + 2 unlabeled), global batch=8 trên 2 GPU. Ghi chú notebook nói mỗi GPU 2 mẫu là sai với mã đã chạy. Đây là yếu tố tăng tải, chưa chứng minh là nguồn gây timeout.

Checkpoint artifact mới nhất đã xác minh metadata: `_latest:v14`, iter_num=3750, kích thước 552975515 byte. Tổng 22 artifact model/checkpoint có dung lượng logic khoảng 11.33 GiB trên cloud; không thể dùng tổng này làm dung lượng cache cục bộ. Theo lịch save local, checkpoint gần nhất dự kiến ở 3950, chưa tải nội dung checkpoint để xác nhận.

## Các run 340–341

Kaggle notebook tài khoản `bokhnhl` hiện trả `CANCEL_ACKNOWLEDGED`. Log output tương ứng run `a35mtk0i`, kết thúc console tại iter 340. Đây là bằng chứng trạng thái hủy của phiên này; không suy ra ai hủy hay nguyên nhân hủy. Không có bằng chứng để quy cả ba run 340–341 cho GPU OOM.

RAM hệ thống cực đại trong events: a35mtk0i=88.33%, fum8iuny=87.63%, 1ex7ty08=87.71%. Không có dấu hiệu RAM chạm 100% trong những mẫu đã thu thập.

## Còn chưa xác định

Vị trí Python/CUDA chính xác nơi rank 0 ngừng tiến triển trước collective. Log ghi FlightRecorder bị tắt nên không có stack collective thất bại. Các hướng cần kiểm tra là DataLoader/HDF5/worker stall, áp lực RAM/I/O, hoặc bất đồng bộ DDP/CUDA. Đây là các giả thuyết, chưa phải kết luận nguyên nhân gốc.

Lần chạy tái hiện nên bật NCCL trace theo thông báo trong log (`TORCH_NCCL_TRACE_BUFFER_SIZE` khác 0), ghi heartbeat theo rank trước/sau lấy batch, forward, backward, optimizer và checkpoint; ghi memory theo cây tiến trình và dung lượng thực của working/cache. Cần thêm stack khi treo để quyết định sửa chính xác. Tăng timeout vượt 30 phút không giải quyết nguyên nhân.

## Dữ liệu lưu kèm

Dữ liệu đã chuyển sang `experiment_data/` và không được commit. Xem
[EXPERIMENT_DATA.md](EXPERIMENT_DATA.md) để tìm snapshot theo run ID, crawl lại
hoặc so sánh offline. Log lỗi nằm trong snapshot Kaggle tài khoản
`thnhnguynvncng`, file `output.log.json` hoặc `output.readable.log`.
Các events, summary và artifact metadata nằm trong snapshot W&B `s9i9nsui`.
Bản sao mã thực thi được lưu dưới `source/` trong snapshot Kaggle tương ứng.

Báo cáo này thay thế các kết luận chưa được chứng minh về SIGKILL/exit 137,
đầy đĩa và memory leak trong `RUN_TERMINATION_AND_MEMORY_ANALYSIS.md`.
