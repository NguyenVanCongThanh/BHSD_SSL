# Kaggle/W&B: dữ liệu có thể lấy và mức hỗ trợ thực tế

Đối chiếu source và môi trường Python ngày **31/10/2026**. Bảng phân biệt provider hỗ trợ, repo có script và portal có implementation. **Portal hiện chưa có implementation**, nên “đã đề cập trong docs” không đồng nghĩa UI/API đã gọi được. Không gọi API có credential trong lần audit này.

Môi trường kiểm tra: `kaggle==1.7.4.5`, `wandb==0.26.1` (metadata package, không login). Đọc source installed CLI cho thấy có files/pull/output, chưa có parser quota và kernels logs. W&B installed source có `heartbeatAt`, `system_metrics`, `scan_history`, `used_artifacts`. Đây là inventory khả năng code, chưa phải kiểm chứng endpoint online. Các venv/container khác có thể khác.

## 1. Kaggle — khả năng và tình trạng repo

Dữ liệu remote chỉ có khi tài khoản được quyền truy cập và provider đã công bố. Notebook đang chạy không đồng nghĩa mọi file trong filesystem container đã tải được.

| Nội dung | Đang chạy | Chạy xong | Repo hiện tại | Portal cần làm |
|---|---|---|---|---|
| Notebook discovery/list/search | Metadata notebook được quyền xem | Tương tự | Executor dùng list để test auth, chưa có chức năng discovery cho người dùng | `list_notebooks`, paging/filter; UI attach/import |
| Status | Status latest session | Terminal status | `kaggle_ctl status`, executor `get_status`; crawler lưu status | Chuẩn hóa result/error, timestamp và version scope |
| Source notebook + metadata | Phiên bản đã lưu, không phải trạng thái RAM editor/container | Phiên bản đã lưu | Installed CLI có pull; controller chưa bọc | `pull_source`, chọn version khi hỗ trợ, digest + diff |
| Console/launcher log | Tùy khả năng công bố của endpoint | Lấy phần log còn được giữ | `kaggle_ctl log`, `crawl_kaggle.py`; không có follow adapter | Log cache, refresh/follow theo capability; xử lý log chưa có/truncated |
| Danh sách output | Chỉ output đã được công bố | File provider còn giữ | `crawl_kaggle.py` lưu tên file, không phải output browser đầy đủ | List có paging, size/version nếu trả về |
| Download output/checkpoint | Không hứa tải file live trong container | Tải file đã publish | Installed CLI hỗ trợ output; controller chưa bọc download chọn lọc | Download job, chọn file/pattern, size limit, checksum, server-managed destination |
| GPU/TPU quota | Theo account nếu CLI/API đã chọn hỗ trợ | Tương tự | Chưa có adapter; CLI installed chưa có quota | Capability-gated quota reader; source/time/unknown, không giả số giờ |
| GPU utilization/VRAM, loss, iteration | Không có trong các tool Kaggle đã audit | Không tự sinh ra từ status | Dựa W&B hoặc file do notebook xuất | Dùng tracking adapter, không gán Kaggle status thành progress |

Nguồn chính thức: [Kernels reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md), [CLI source](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/cli.py), [Quota reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/quota.md). Reference hiện có `logs --follow`, `quota` và các push options bổ sung; **không suy chúng có trong CLI installed hoặc controller BHSD**. Nâng version cần compatibility test trước khi bật UI. Quota là accelerator quota, không phải billing/storage quota.

`get_log()` dùng REST output, crawler gọi generated/internal client để lấy session output. Đây không phải wrapper ổn định đã được kiểm chứng cho mọi phiên bản. Giữ raw provenance, chặn signed URL/secret khỏi snapshot. `get_status()` hiện trả chuỗi có thể lẫn stderr; cần typed adapter trước khi dùng state machine.

## 2. W&B — khả năng và tình trạng repo

Chỉ lấy được dữ liệu đã log/upload/sync. Online có độ trễ; offline chưa sync thì cloud API không có. Run finished không bảo đảm upload đủ khi process crash hoặc mạng lỗi.

| Nội dung | Đang chạy | Chạy xong | Repo hiện tại | Portal cần làm |
|---|---|---|---|---|
| Run list/identity/state/URL | Có theo quyền | Có theo retention | `fetch_wandb`, `inspect_wandb_runs`, `fetch_latest_run` | Multi-connection, project/filter/paging; bỏ hardcode |
| Config, summary | Giá trị đã đồng bộ, summary còn thay đổi | Giá trị cuối đã đồng bộ | `crawl_run`, exporters | Config provenance, snapshot versions; summary không mặc định là best |
| Loss/LR/iteration/validation history | Các rows đã server nhận | History còn được giữ | `scan_history` trong crawler/exporter; fetch_wandb chọn validation milestones | Incremental cursor, overlap/dedupe, exact export tách sampled chart |
| System telemetry CPU/RAM/GPU/VRAM | Nếu SDK đã thu và sync | History đã lưu | `crawl_run` dùng `history(stream='events', samples=10000)` | Latest `system_metrics`, timestamps; events có giới hạn mẫu, không gắn nhãn full |
| W&B heartbeat | Tín hiệu tracking process/server | Heartbeat cuối, không còn là liveness | SDK installed có heartbeat field; crawler chưa export riêng | Normalize heartbeat nếu SDK trả; tách heartbeat/progress/provider status |
| Console output + runtime metadata | Tệp đã upload, có thể thiếu/đang thay đổi | Tệp còn giữ | `crawl_run` chỉ tải `output.log`, `wandb-metadata.json` | File catalog và download policy, freshness/truncation |
| Image/mask/Table/media | Chỉ media đã log và upload | Media đã giữ lại | Training có logging, crawler không tải media tổng quát | Media manifest + lazy download/render; không quét tải tất cả |
| Output artifact/checkpoint | Chỉ version đã commit/upload xong | Version còn tồn tại | Crawler list metadata; `resume.py` và manager tải checkpoint cho luồng resume | Browse/version/digest/manifest/download tổng quát |
| Input artifact lineage | Chỉ input được khai báo sử dụng | Quan hệ đã khai báo | Chưa thấy exporter dùng `used_artifacts()` | Export input/output lineage; dữ liệu không khai báo phải unknown |
| Tags/notes/group/job type/code commit | Khi có metadata | Tương tự | Chưa export đầy đủ các field | Normalize fields, không coi commit riêng là toàn bộ code provenance |

Nguồn chính thức: [W&B Run implementation](https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py). History và artifact APIs không tạo dữ liệu mà notebook chưa log. Input lineage cần explicit artifact use; heartbeat chỉ là tín hiệu tracking, không chứng minh training loop tiến triển.

`scan_history(keys=[...])` có thể chỉ trả rows chứa đủ các keys yêu cầu; metrics log khác bước cần query riêng hoặc merge phù hợp. Không mặc định dùng `_step` thay training iteration sau resume. Không gọi full `scan_history()` mỗi phút cho mọi run. Snapshot `history.json` phản ánh dữ liệu truy vấn tại thời điểm crawl, không tự là lịch sử hoàn chỉnh vĩnh viễn.

## 3. Những thông tin không thể tự suy ra

- ETA, tiến độ %, GPU-hours theo attempt: tính từ timestamp/iteration/budget đã xác minh; gắn estimated. Runtime W&B không mặc định là GPU billing time.
- Nguyên nhân crash: đối chiếu launcher log, tracking log, status và timestamps; không có log thì unknown, không đoán OOM/time limit từ im lặng.
- Best model: phải nêu metric/protocol và checkpoint thực tồn tại. Summary tốt nhất không tự chứng minh checkpoint tương ứng đã upload.
- Prediction volume/GT: phải có artifact/dataset và quyền truy cập; ảnh RGB trên W&B không thay thế tensor/mask 3D. Không thể phục hồi volume từ vài ảnh preview.
- Process trong Kaggle, RAM filesystem chưa publish, arbitrary SSH/container control: chưa có tool/contract trong phạm vi audit để cung cấp các chức năng này.

## 4. Docs portal đã bao phủ đến đâu?

Trước audit này, requirements/architecture có nhắc metrics, system telemetry, logs, artifact lineage, snapshots và account quota. Operations spec đặc tả status/heartbeat/submit. **Chưa có capability matrix theo provider/version, API output browser/download/media, export input lineage hay quy trình final reconciliation đầy đủ.** Vì vậy không thể kết luận docs đã sử dụng hết tool hoặc mọi capability của provider.

Không cần tích hợp mọi API Kaggle/W&B. Phạm vi là notebook/training observability và artifacts; competition submissions, forum, billing/admin, sweep orchestration và xóa remote nằm ngoài MVP hiện tại.

### Adapter contract cần thêm

Các adapter trả payload có schema version + connection identity + remote reference/version + fetched_at + source + completeness/freshness + cursor + warnings. Missing/forbidden/not-yet-published/unsupported/rate-limited phải phân biệt, không đổi thành empty list hoặc số 0.

**Kaggle:** `capabilities`, `list_notebooks`, `get_status`, `pull_source`, `fetch_logs`, `list_outputs`, `download_outputs`, `get_quota`; build/submit tiếp tục theo operations spec.

**W&B:** `capabilities`, `list_runs`, `get_run`, `get_progress`, `get_system_metrics`, `fetch_history`, `fetch_system_events`, `list_files`, `download_file`, `list_logged_artifacts`, `list_used_artifacts`, `get_artifact_manifest`, `download_artifact`.

Capability discovery là version inventory + local help/source check + authenticated read probe có kiểm soát. Kết quả theo connection/endpoint; thấy lệnh trong help không chứng minh account có quyền. Mọi download chạy qua worker, allowlist local root, giới hạn dung lượng và không expose signed URLs/API keys.

### API/UI bổ sung (đề xuất)

- Account detail: capability/version panel và quota (nếu supported).
- Notebook detail: Source/metadata, Outputs (list trước download), Logs; file/version scope rõ ràng.
- Run detail: Config/Summary, Curves, System, Files/Media, Artifacts in/out.
- `GET /api/kaggle/accounts/{id}/capabilities`, `/quota`; notebook `/source`, `/outputs`; `POST .../downloads` tạo job.
- `GET /api/wandb/runs` và `/api/wandb/runs/{id}`; các resource `/history`, `/system`, `/files`, `/artifacts` dùng remote full reference từ DB; `POST .../downloads` và `/exports` tạo job. Quyền kiểm tra ở backend, không dựa vào UI.

### Đang chạy và khi kết thúc

- **Active:** worker đọc status/summary/progress định kỳ, history incremental với overlap và dedupe; system theo cadence phù hợp; logs/media/artifact manifests tải khi cần. Poll chậm hơn/backoff khi lỗi.
- **Terminal observed:** enqueue final reconciliation cho status, metadata, summary/history, logs và artifact/output inventory. Upload có thể còn đang hoàn tất: retry có backoff, giới hạn số lần/thời gian cấu hình được; lưu `pending/partial/complete/unknown` và lý do. “Complete” chỉ đối với tập tài nguyên đã yêu cầu, không có nghĩa thu mọi dữ liệu provider từng có. Cho resync thủ công.
- Ghim source version/artifact digest khi xuất báo cáo; raw events sampled phải có nhãn sampled. Lưu final snapshot bất biến, không dùng alias latest làm bằng chứng duy nhất.

### Thứ tự bổ sung và nghiệm thu

1. Accounts + typed status/log/progress và capability inventory; không chặn monitor khi quota unsupported.
2. Notebook discovery/source/output browser/download chọn lọc; multi-account isolation và version scope được test.
3. W&B incremental history + latest system + heartbeat adapter; phân biệt sampled/missing/stale và iteration đứng.
4. Media/file/artifact browser + input/output lineage + final reconciliation; không bắt tất cả download mới cho xem dashboard.

Nghiệm thu với fixtures và provider read probes được cấp quyền: không lẫn identity; không full-scan mỗi poll; file chưa publish khác file rỗng; final sync thiếu upload hiển thị partial; quota/log-follow unsupported có fallback rõ; download/checkpoint ghim version; manifest không chứa secret/signed URL. Test offline không thay cho xác minh online API compatibility.
