# Đặc tả Kaggle Operations MVP

> Điểm bắt đầu hiện hành: [PoC-first](poc-first.md) quy định stack/phạm vi mặc định. Full stack, hosting và REST API trong tài liệu này là hướng mở rộng; không yêu cầu triển khai trước PoC.

Đối chiếu `tools/kaggle/kaggle_ctl.py`, `accounts_manager.py`, `notebook_builder.py` và `kaggle_executor.py` ngày **31/10/2026**. Đây là yêu cầu triển khai và nghiệm thu, chưa phải tính năng đã có. Bổ sung [platform review](platform-review-2026-10-31.md). Với ưu tiên người dùng vừa xác định, **Account UI + notebook monitor + submit form là phạm vi bản usable đầu tiên**, không cần chờ hoàn thiện toàn bộ research registry/viewer.

## 1. Luồng và màn hình bắt buộc

1. **Accounts**: liệt kê → thêm/sửa → test kết nối → chọn mặc định hoặc disable.
2. **Notebooks**: thêm notebook đang có bằng `owner/slug` → chọn connection quan sát → check ngay hoặc auto-refresh → xem status/heartbeat/log/link Kaggle.
3. **Submit**: chọn account + template → chỉnh options → build/validate → xem preview → submit → theo dõi operation và notebook vừa gửi.

Có thể triển khai thành ba tab trong một giao diện. UI lưu filter/account đang chọn nhưng không lưu token vào browser storage. Không bắt người dùng sửa JSON hoặc gõ CLI cho các luồng thông thường.

## 2. Account CRUD trên UI

### Form và quy tắc

| Field | Quy tắc |
|---|---|
| `id` | ID nội bộ bất biến, server cấp; giữ legacy ID riêng khi import |
| `name`, `slug`, `description` | Tên dễ nhớ, slug duy nhất không phân biệt hoa thường, mô tả tùy chọn |
| `username` | Kaggle principal dự kiến; so sánh với principal khi test nếu provider cho phép xác minh |
| `auth_type` | Loại credential adapter thực sự hỗ trợ; không hiện OAuth khi chưa triển khai flow |
| `credential` | Input write-only; chỉ nhập khi thêm/thay; sửa metadata không yêu cầu nhập lại |
| `enabled`, `is_default` | Account disable không dùng cho submit mới; default phải enabled |
| `revision` | Optimistic concurrency: sửa trên revision cũ trả conflict, không ghi đè im lặng |

Khi mở Edit, token cũ không được tải về UI. Field secret trống nghĩa là giữ nguyên, không phải xóa credential. Thay credential là thao tác tường minh; UI chỉ hiển thị configured/last rotated/last tested, không hiển thị cả prefix token.

- Nút **Save** cho phép lưu draft chưa test nhưng không đánh dấu verified. **Test connection** chạy job có timeout, trả `verified`, `invalid_credentials`, `forbidden`, `network_error`, `unknown` và thời điểm kiểm tra. Lỗi mạng không đồng nghĩa key sai.
- Username là identity: khi đã có lịch sử execution, thay username phải tạo profile mới hoặc bị chặn; sửa tên hiển thị/slug không đổi lineage. Không tái sử dụng cùng account ID cho người khác.
- Không chọn default bằng fallback âm thầm nếu account đã chọn thiếu credential. Import account cũ chỉ nhập metadata trước, credential được di chuyển vào secret store bằng thao tác riêng; không trả key về browser.
- Disable giữ nguyên history và cho xem dữ liệu cache. Nếu credential vẫn còn, có thể tiếp tục monitor các attempt hiện có theo lựa chọn rõ ràng. Disconnect/revoke ngừng gọi provider và hiển thị cached/stale.
- Xóa vĩnh viễn chỉ áp dụng profile chưa có references/job; profile đã dùng thì archive. Đổi secret/disable không đổi account của các job đã tạo. Job queued bị disable trước dispatch phải dừng và yêu cầu chọn lại; job đã dispatch giữ lineage và không tự submit lại.
- Secret store + metadata update phải có cơ chế rollback/recovery khi một bên ghi lỗi; log audit không chứa secret.

Không gọi `account add --key ...` từ web server: token trong command argument có thể lọt process list/log. Adapter nhận secret qua kênh nội bộ/env/file riêng; API account không trả secrets. Không gọi trực tiếp phương thức ghi plaintext config của manager hiện tại làm storage cho web.

## 3. Notebook status và heartbeat

### Ba nguồn tín hiệu khác nhau

| Tín hiệu | Trả lời câu hỏi gì? | Mức bảo đảm |
|---|---|---|
| Kaggle status | Provider gần nhất báo queued/running/complete/error gì? | Không chứng minh training còn tiến triển |
| W&B telemetry/iteration | Lần cuối nhận telemetry và lần cuối iteration tăng khi nào? | Cần bind đúng run/attempt; không có W&B thì unavailable |
| Heartbeat notebook tùy chọn | Code notebook lần cuối phát liveness signal khi nào? | Cần instrumentation; thread heartbeat sống không chứng minh training loop tiến triển |

MVP bắt buộc có polling Kaggle status, thời điểm kiểm tra và stale indicator. Nếu liên kết được W&B thì hiển thị thêm last telemetry, last progress và iteration. **Không được gọi polling API thành bằng chứng training còn sống.** Heartbeat chủ động về portal là tính năng tùy chọn sau, cần endpoint có thể truy cập từ Kaggle và credential riêng theo attempt; không phải điều kiện để dùng monitor qua W&B.

Mỗi notebook/attempt có:

- `account_id`, `owner/slug`, remote version nếu xác định được, attempt ID, W&B full reference nếu có.
- `provider_state_raw`, `provider_state_normalized`, `provider_observed_at`, `last_check_attempt_at`, `last_check_error`.
- `last_telemetry_at`, `last_progress_at`, `last_iteration`, `heartbeat_source`, `source_available`.
- `health`: `pending`, `progressing`, `suspected_stall`, `terminal`, `unknown`; kèm evidence và freshness. `progressing` chỉ khi có iteration tăng trong khoảng quan sát, không suy từ RUNNING.

Mặc định cấu hình: poll status active mỗi 60 giây, đánh dấu status stale sau 3 chu kỳ không thành công; nghi stall khi RUNNING và không thấy progress >20 phút. Đây là ngưỡng cảnh báo có thể chỉnh theo cadence validation/startup; không phải giới hạn Kaggle. W&B system telemetry và progress timestamp tách riêng. Khi mới attach, không biết history hoặc chưa có tín hiệu thì hiện unknown/pending; không gán failed.

- Nút **Check now** enqueue job và có cooldown, không gọi provider vô hạn khi bấm liên tục. Worker backoff/jitter khi lỗi/429.
- `RUNNING + progress mới`: “Đang chạy — tiến độ cập nhật … trước”.
- `RUNNING + progress cũ`: “Kaggle báo running — nghi ngờ không tiến triển”; không tự cancel/resubmit.
- Provider query lỗi: “Không kiểm tra được — trạng thái gần nhất … lúc …”; không đổi sang ERROR của notebook.
- Provider terminal nhưng telemetry gần đây: giữ cả hai nguồn và thời điểm; không tự đoán thứ tự do đồng hồ/cache.
- Log có thể chưa được provider công bố; hiện “chưa có log”, không coi là notebook chết. Không hứa live stdout nếu endpoint chỉ có output sau session.
- Khi slug đã có version mới, không gán status slug hiện tại cho attempt cũ một cách chắc chắn; ghi latest-version/unverified khi không query được version cụ thể.

Không có W&B vẫn dùng Accounts, Submit và Kaggle status bình thường. Không yêu cầu sửa mọi notebook chỉ để monitor cơ bản.

## 4. Ánh xạ form Submit với flags đang có

Form lấy capability/defaults từ adapter theo revision. Bảng dưới là baseline CLI hiện tại; UI không hardcode mọi template đều hỗ trợ resume.

| Field trên UI | CLI tương ứng | Quy tắc |
|---|---|---|
| Account | `push --account/-a` | Bắt buộc xác nhận account cụ thể; build preview dùng username của account |
| Template | `--notebook/-n` | Từ notebooks config; mặc định CLI `mean_teacher` |
| Task | `--task/-t` | `multiclass` hoặc `binary`; mặc định CLI multiclass |
| Kernel slug | `--slug` | Optional; preview full `owner/slug` sau resolve |
| Title | `--title` | Optional; hiển thị title thực sau resolve |
| Dataset source override | `--dataset/-d` | CLI hiện nhận một source; trống dùng template; preview liệt kê toàn bộ source đã resolve |
| W&B connection | `--wandb-key/-w` về ngữ nghĩa credential | Chọn connection hoặc Kaggle Secrets/offline; không đưa literal key vào command/preview |
| Enable GPU | đảo giá trị `--no-gpu` | Bật GPU mặc định; loại GPU/TPU chỉ hiện nếu adapter hỗ trợ, chưa phải flag hiện có |
| Resume run | `--resume-run` | ID hoặc `entity/project/id`; UI ưu tiên full reference để tránh nhầm project |
| Resume tracking | `--resume-mode` | `same-run`/`new-run`; mặc định same-run khi resume |
| Checkpoint policy | `--resume-checkpoint` | `latest`/`best`; mặc định latest khi resume |

`build --username/-u` là target metadata cho dry run, không phải credential hay quyền submit. UI submit lấy owner từ account, không cho override để giả làm account khác. Các field account editor tương ứng `account add --id/--username/--key/--name/--slug/--desc` về chức năng; dùng API CRUD với secret write-only thay vì lặp nguyên giao diện CLI.

Resume mode/checkpoint đòi hỏi resume run. Builder hiện chỉ hỗ trợ resume cho `mean_teacher`; form phải disable và giải thích ở template khác. Bare run ID phải resolve theo W&B project được chọn rồi hiển thị full reference. Checkpoint `best` phục vụ lựa chọn của người dùng; không mặc định coi là checkpoint tốt nhất để tiếp tục training.

Không có trường “extra flags” tự do chạy shell. Các options ngoài CLI hiện tại (tùy ý config YAML, seed/LR/max iteration, GPU model, nhiều dataset override, stop/cancel) phải được ghi unsupported hoặc triển khai adapter riêng trước khi bật UI. Có thể bổ sung config editor sau nhưng phải qua parser cùng revision và lưu resolved config; không giả vờ mọi key đã được inject vào notebook.

### Build → Preview → Submit

1. **Build/validate** chạy local qua worker, workspace riêng. Trả config/task/template đã resolve, account principal, dataset list, GPU enabled, resume settings, notebook/metadata digest và validation messages. Không có side effect push.
2. Preview cho xem metadata và notebook đã loại secret; nút download chỉ xuất bản an toàn. Command preview che secret bằng reference, không chứa token.
3. **Submit** dùng `preview_id`, request hash và idempotency key. Nếu account credential revision, template/source hoặc options thay đổi sau preview, buộc build lại trước dispatch.
4. Tạo operation bền vững, worker claim và push bundle đã preview. Không build lại một bundle khác sau khi đã chấp nhận preview. Không tự chọn account khác khi fail.
5. Trả operation/attempt, kernel URL và remote version nếu có. Tự thêm notebook vào monitor.
6. Push timeout mà chưa biết provider đã nhận chưa: `reconciling/unknown`; không tự retry push. Worker kiểm tra remote evidence; nếu không đủ thì yêu cầu operator xử lý trạng thái bất định trước lần submit tiếp.

Không đồng nhất “request được nhận”, “push thành công”, “notebook running” và “training thành công”. Bấm Submit lần hai với cùng idempotency key phải trả operation cũ; cùng key khác payload trả conflict. Khóa theo connection+owner/slug chống hai job đẩy đồng thời.

## 5. API contract tối thiểu (đề xuất, chưa có implementation)

Các endpoint có scope project/account và authorization tại backend. Job API trả operation ID và không giữ HTTP request chờ provider lâu.

| Endpoint | Kết quả chính |
|---|---|
| `GET/POST /api/kaggle/accounts` | List metadata / tạo profile; POST secret write-only |
| `PATCH /api/kaggle/accounts/{id}` | Sửa metadata với revision |
| `PUT /api/kaggle/accounts/{id}/credential` | Thay secret, invalidate verification |
| `POST /api/kaggle/accounts/{id}/test` | Enqueue test connection |
| `DELETE /api/kaggle/accounts/{id}` | Xóa profile chưa dùng; conflict nếu có references |
| `GET /api/kaggle/templates` | Template options/defaults/capabilities và adapter revision |
| `GET/POST /api/kaggle/notebooks` | List / attach notebook có sẵn với account connection |
| `GET /api/kaggle/notebooks/{id}` | State, health, timestamps, evidence |
| `POST /api/kaggle/notebooks/{id}/check` | Enqueue check ngay |
| `GET /api/kaggle/notebooks/{id}/logs` | Log cache đã redacted, freshness và availability |
| `POST /api/kaggle/notebooks/{id}/fetch-logs` | Enqueue lấy log, giới hạn kích thước |
| `POST /api/kaggle/previews` | Enqueue build/validate từ typed options |
| `GET /api/kaggle/previews/{id}` | Preview metadata/digest, validation, expiry |
| `POST /api/kaggle/submissions` | Submit preview, bắt buộc idempotency key |
| `GET /api/operations/{id}` | queued/running/succeeded/failed/reconciling + outcome |

Error có mã máy đọc được, message đã sanitized và retryability. Không trả raw stderr/HTTP body chưa redacted. Default account thay qua account update trong transaction để duy trì duy nhất một default theo scope. Log output path là server-managed, không nhận đường dẫn tùy ý từ browser (`log --output` CLI không chuyển nguyên sang API). UI có tùy chọn tail tương đương `log --tail`, bị giới hạn server-side. Chọn notebook bằng full kernel reference tương đương `status/log --kernel`; account vẫn là credential dùng để truy cập.

## 6. Checklist nghiệm thu để giao triển khai

- Thêm account, sửa name/description, đổi credential và test được hoàn toàn trên UI; reload vẫn giữ metadata. Field secret trống khi edit giữ key cũ. Hai người sửa đồng thời nhận conflict thay vì mất cập nhật.
- Hai account có credential khác nhau không dùng nhầm token khi build/check/submit song song; credential lỗi không fallback account máy chủ. API/log/notebook tải về không chứa token hoặc prefix token.
- Disable chặn submit mới, không mất history. Lỗi auth/network phân biệt được; đổi credential buộc xác minh lại.
- Attach được notebook có sẵn; Check now và auto-poll có timestamp/freshness; mất mạng không biến notebook thành failed. Chạy được khi không có W&B.
- Có W&B: iteration mới cập nhật progress; chỉ telemetry mới không che tình trạng iteration đứng; ngưỡng stall cấu hình được. RUNNING không telemetry hiện unavailable/unknown.
- Mỗi flag trong bảng Submit có control tương ứng hoặc lời giải thích unsupported. Kiểm tra defaults/overrides trên metadata và notebook thực, không chỉ form payload. Resume thiếu run/sai template bị từ chối trước push.
- Build không push. Submit đúng bundle digest/account/options đã preview; form thay đổi làm preview cũ mất hiệu lực. Không đưa secrets vào command args.
- Double click/retry không push trùng; provider timeout sau nhận request không auto-submit lại; restart worker phục hồi operation mà không dispatch mù lần hai.
- Submit thành công tự xuất hiện trong monitor, có URL và lineage; version không xác minh được ghi unknown. Provider status, progress, sync health và kết quả training hiển thị riêng.

Các adapter cần chỉnh trước khi bọc API: manager lưu profile/secret và fallback, CLI in prefix W&B key và thay env toàn cục, builder dùng output path chung, executor status trả chuỗi trộn lỗi/trạng thái. Không coi các hàm này đã an toàn cho web concurrency chỉ vì CLI dùng được.

## 7. Dữ liệu provider và capability theo phiên bản

Xem [capability audit](provider-capabilities.md) cho source/output/download, W&B files/media/artifact lineage, quota và final reconciliation. CLI Kaggle mới có thể hỗ trợ logs-follow/quota nhưng môi trường audit chưa có; bật UI theo capabilities đã kiểm chứng.
