# Lưu và crawl kết quả thí nghiệm

Code và báo cáo nằm trong Git. Dữ liệu crawl nằm ở `experiment_data/`, đã được
`.gitignore` loại trừ. Không xóa thư mục này nếu còn cần so sánh offline; muốn dọn
repo thêm có thể chuyển nó sang ổ lưu trữ khác và đặt `BHSD_EXPERIMENT_DATA` trỏ
đến đường dẫn mới. Đây là dữ liệu local, không tự backup khi commit/push.

```text
experiment_data/
  wandb/<run_id>/<timestamp_UTC_unique>/
    manifest.json
    summary.json
    config.json
    events.json
    history.json
    output.log
    artifacts.json
  wandb/_collections/<timestamp_UTC_unique>/wandb_results.json
  kaggle/<owner>/<kernel_slug>/<timestamp_UTC_unique>/
    manifest.json
    output.log.json
    status.json
    files.json
  imports/migration/<timestamp_UTC_unique>/files.json
```

File có mặt tùy loại crawl; script tải log riêng không tạo history/config.
Manifest ghi ID và thời điểm crawl UTC. Bản nhập từ dữ liệu cũ có thời điểm crawl
`null`, vì chỉ biết lúc di chuyển. Không diễn giải thời điểm di chuyển thành thời
điểm crawl. Snapshot không ghi đè; dữ liệu Kaggle được lưu theo thời điểm lấy và
ghi rõ khi API không cung cấp version ID. Nên đối chiếu tên/run ID trong log để
ghép W&B với đúng phiên Kaggle.

## Crawl lại

Dùng môi trường có `wandb`/`kaggle` tương ứng. W&B sử dụng `WANDB_API_KEY` hoặc
credentials từ `wandb login`; các script đã cập nhật không ghi đè bằng key trong
mã. Kaggle dùng credentials hiện có hoặc chọn account đã cấu hình.

```bash
python tools/crawl_run.py ENTITY/PROJECT/RUN_ID
python tools/crawl_kaggle.py OWNER/KERNEL_SLUG --account 1
python tools/fetch_latest_run.py
python tools/fetch_wandb.py
python tools/list_runs_summary.py
```

`crawl_run.py` lưu history, summary, config, danh sách artifact và system events
(tối đa 10000 mẫu qua API history). Nó chỉ tải log/metadata, không tải checkpoint.
Log W&B có thể thiếu lỗi launcher, nên lấy thêm log Kaggle khi phân tích crash.
`crawl_kaggle.py` chỉ lấy trạng thái, log và tên file; không tải model hoặc lưu URL
download có token. Muốn tái hiện phiên cũ cần lưu mã/commit tương ứng trước khi
output Kaggle bị thay thế; W&B/Kaggle không bảo đảm tải lại mãi được.

## So sánh offline

```bash
python tools/compare_runs.py --input experiment_data/wandb/_collections/SNAPSHOT/wandb_results.json
python tools/compare_trajectories.py --input experiment_data/wandb/_collections/SNAPSHOT/wandb_results.json
python tools/check_per_class_trend.py --input experiment_data/wandb/_collections/SNAPSHOT/wandb_results.json
```

Nếu bỏ `--input`, các script chọn collection snapshot mới nhất có file tương ứng.
Các script này dùng schema collection của `fetch_wandb.py`, không phải summary
đơn lẻ từ `crawl_run.py`.

## Dữ liệu đã dọn

Đã chuyển audit, JSON kết quả ở gốc/docs và log trong tools vào cấu trúc trên.
Giữ log Kaggle nguyên bản, events theo run, summary, metadata, artifact metadata
và bản mã đã chạy để đối chiếu lịch sử. Các log trùng được gộp sau khi kiểm tra
hash; system_events trùng được kiểm tra nội dung JSON. Bản readable log được giữ
vì khác cách biểu diễn xuống dòng. Manifest migration có inventory SHA-256.

Báo cáo sự cố: [RUN_TERMINATION_AUDIT.md](RUN_TERMINATION_AUDIT.md).
Xóa snapshot nào sẽ mất bản so sánh local của snapshot đó; xóa dữ liệu local
không xóa run, artifact hay checkpoint trên W&B/Kaggle.
