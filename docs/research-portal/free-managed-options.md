# Phương án miễn phí, managed services và UI có sẵn

> Điểm bắt đầu hiện hành: [PoC-first](poc-first.md) quy định stack/phạm vi mặc định. Full stack, hosting và REST API trong tài liệu này là hướng mở rộng; không yêu cầu triển khai trước PoC.

Khảo sát tài liệu chính thức ngày **31/10/2026**; chưa tạo tài khoản hay deploy/probe Kaggle/W&B trên các runtime này. Free tier có thể thay đổi, không phải bảo đảm vận hành 24/7. Đây là lựa chọn thay thế service VM trong migration-and-hosting, không phải bắt buộc mua/cấp VM.

## 1. Điều chỉnh kiến trúc để không cần VM thường trực

UI mở bằng URL; dữ liệu/operations bền vững ở hosted DB; API ngắn chạy khi được gọi; scheduler kích hoạt worker hữu hạn, đọc job và cập nhật snapshot rồi thoát. Không cần một process Python ngồi `while True` cả ngày. Vì vậy poll khi browser đóng vẫn có thể thực hiện bằng scheduled tasks, trong giới hạn của nền tảng. Điều này khác host một web app bị sleep rồi trông chờ thread nền tiếp tục chạy.

Các yêu cầu domain không đổi: credential isolation, idempotency, ambiguous submit reconciliation, raw provenance, freshness và version scope. Scheduled runner retry không được tự push notebook lần hai khi lần trước có thể đã được nhận.

## 2. Các lựa chọn có căn cứ

| Nền tảng | Vai trò khả thi | Free tier và giới hạn cần biết | Đánh giá cho portal |
|---|---|---|---|
| Supabase | Postgres/Auth/Storage, API CRUD, Cron/Edge Functions | Free có DB 500 MB, storage 1 GB, có pause khi ít activity; functions free walltime 150s, CPU 2s | Hợp metadata/auth; không thay Python CLI worker nguyên trạng |
| Streamlit Community Cloud | UI Python, gọi API hoặc đọc DB server-side | Deploy từ GitHub, miễn phí; app idle 12h sẽ sleep | Dễ nhất để làm UI bằng Python; polling độc lập phải ở runner khác |
| Appsmith Cloud | UI kéo thả: accounts table/form, monitor, submit buttons | Free cloud tối đa 5 users theo pricing khảo sát; Workflows nằm Business | Hợp operations UI nhanh; custom provider adapter và scheduler vẫn cần riêng |
| Trigger.dev Cloud | Task runner/scheduler, có Python extension | Free $5 credits/tháng, 10 schedules, 1 ngày log retention; hết credit cần nâng cấp để tiếp tục | Candidate cho Python Kaggle/W&B jobs; TypeScript task wrapper + Python dependencies cần PoC |
| Cloudflare Pages | Host React static UI | Static requests miễn phí theo pricing; functions có limits riêng | Hợp UI lâu dài; không chạy CLI Python bằng static hosting |
| Render Free | Host Python API/web app | Idle 15 phút sleep, filesystem ephemeral; free Postgres hết hạn sau 30 ngày | Demo/on-demand API; không chọn làm worker/DB thường trực miễn phí |
| Windmill self-host | Forms/apps + Python scripts + schedules | Free/open-source cần tự host; cloud không được suy là free | Phần mềm khá gần nhu cầu, nhưng không giải quyết thiếu máy host |
| Oracle Always Free | VM để chạy stack hiện tại | Có giới hạn/capacity, tài nguyên idle có thể bị thu hồi | Hợp nếu muốn giữ FastAPI/worker, nhưng vẫn phải quản trị VM |

Không chọn Pipedream theo giả định free production vì pricing chính thức đang phân biệt Free development và paid production. Hugging Face Spaces cũng không đưa vào shortlist worker: docs hiện nêu điều kiện paid plan cho compute Spaces dù CPU Basic hardware có mức FREE; có sleep và disk không persistent. Phải phân biệt hosting eligibility với hourly hardware price.

## 3. Hai cấu hình đề xuất

### A. Dễ làm và giữ Python: Streamlit + Supabase, thêm Trigger.dev khi cần lịch nền

- Streamlit UI: Accounts/Notebooks/Submit; server-side access, viewer restriction và per-user/project permissions.
- Supabase: profile metadata, runs/observations, notes, operations và cursors. Auth/RLS tùy mô hình truy cập; không cấp service-role key cho browser.
- Credential nhập trên UI qua authenticated server endpoint, lưu Vault hoặc encrypted secret store với permission riêng; DB browser-facing chỉ thấy credential reference. Không expose Vault decrypted view qua public API.
- Trigger.dev: những task như `check_notebooks`, `sync_wandb`, `build_preview`, `submit_bundle`. Ban đầu có thể chỉ refresh thủ công, nhưng không hứa polling khi UI sleep nếu chưa có scheduler độc lập.
- W&B/Kaggle tiếp tục giữ checkpoints; Supabase chỉ giữ manifest, metric summary/validation milestones và ít preview, không mirror toàn history/checkpoint lớn.

Đây là đề xuất, chưa chứng minh adapters hiện tại chạy được nguyên trạng trên Trigger.dev. Cần đóng gói BHSD templates/code theo revision, sửa global env/path, và đo import/runtime/memory trước.

### B. Ít viết UI: Appsmith + Supabase + runner

Appsmith cung cấp table/form/button/chart; API server-side/Edge Functions kiểm tra quyền, ghi operations và gọi runner. Không mặc định Appsmith Free có background Workflows. Không đưa Kaggle/W&B token vào client-side JavaScript/query parameter. Có thể dùng builder để tạo màn hình trước, nhưng vẫn cần adapter custom cho flags, heartbeat và lineage.

Nếu muốn UI chuyên biệt/viewer BHSD lâu dài, thay Appsmith/Streamlit bằng React static trên Pages; data/worker contracts vẫn giữ. Không cần triển khai cả ba loại frontend.

## 4. Supabase làm được bao nhiêu?

Supabase đủ để thay DB server, auth và phần API CRUD. Cron + pg_net có thể gọi Edge Functions định kỳ; Vault dùng lưu secret được kiểm soát quyền. Edge Functions là Deno/TypeScript, không phải môi trường chạy trực tiếp `kaggle_ctl.py`/W&B Python SDK. Với read-only API HTTP ngắn có thể viết adapter TypeScript; không tự chuyển internal endpoints thành stable contract. Build notebook, gọi CLI và scan history dài nên giao Python runner.

Nếu không cần giữ Python và chỉ lấy status/summary nhỏ, có thể thử Supabase Cron+Functions không cần runner thứ ba. Đây là phương án cần PoC API compatibility riêng. Không cố nhét full history export, checkpoint download hay build nặng vào function bị giới hạn CPU.

## 5. Chi phí và nhịp polling

Không hứa toàn bộ workload luôn $0. Ví dụ tính toán theo bảng Trigger.dev khảo sát: một scheduled batch task mỗi 5 phút trong 30 ngày có 8.640 runs. Nếu mỗi task thực thi 10s trên small-1x, compute+invocation khoảng $3,14; mỗi phút khoảng $15,68. Đây là phép tính minh họa, chưa tính tasks con/retries/download/build và thời gian cài/import; không phải báo giá hoặc benchmark. Batch nhiều account trong cùng task khác tạo một task mỗi notebook. Ghi observed_at và cho Check now; không dùng lịch thưa làm real-time heartbeat.

Bắt đầu poll active notebooks mỗi 5–10 phút, terminal ít hơn; giới hạn concurrency và export full history theo yêu cầu. Kho dữ liệu ít MB metadata hợp free tier hơn checkpoint vài trăm MB. Nếu hết credits/scheduler pause, UI phải báo stale, không ghi notebook failed. Persist operation outcomes trong DB vì runner log retention có thể ngắn.

## 6. Trải nghiệm cài đặt và sử dụng

Thiết lập một lần: Git repo → tạo hosted DB/schema/permissions → deploy UI → cấu hình runtime secrets → deploy task runner và schedule → kiểm tra hai account, duplicate submit, timeout và network errors. Không cần SSH tunnel cho các thao tác thuần Kaggle/W&B; phát triển repo trên HPC rồi push vẫn được.

Sau thiết lập, người dùng mở URL, đăng nhập, thêm/sửa account, test, attach notebook, xem progress, build preview và submit. Không cần mở console của ba nhà cung cấp mỗi lần sử dụng. Hiện chưa có template turnkey đã được audit trong khảo sát này để bảo đảm toàn bộ luồng Kaggle đa tài khoản + W&B + resume hoạt động ngay mà không viết integration.

Cloud UI không tự đọc filesystem/private network HPC. MVP Kaggle/W&B có thể độc lập HPC; local viewer/evaluation/import snapshot cần sync có chọn lọc hoặc agent được HPC cho phép. Không chuyển cả dataset ra cloud chỉ vì dùng Supabase.

## 7. PoC trước khi chọn mặc định

1. Deploy private UI và DB metadata, credential write-only với backend authorization.
2. Runner probe status một notebook và config/summary một W&B run bằng identity tường minh; không in secret.
3. Scheduled task cập nhật khi browser đóng; UI sleep/wake không mất dữ liệu.
4. Build dry-run đúng template/revision, không push; đo runtime/credit/memory.
5. Chỉ sau đó thử submit do người dùng yêu cầu; idempotency và timeout reconciliation như operations spec. Không cấu hình retry push mù.
6. Kiểm tra free eligibility, secret access, pause/credit exhaustion và backup/export; quyết định giữ free hay cần đổi nền tảng.

Không đổi cấu trúc repo thành phụ thuộc vendor toàn bộ: giữ packages/contracts và adapters; thêm `supabase/{migrations,functions}` nếu chọn Supabase, `tasks/trigger` cho runner, `apps/streamlit` hoặc `apps/web` cho frontend được chọn. `services/api/worker` là vai trò logic, có thể được hiện thực bằng functions/tasks thay vì hai daemon. Đây là deployment profile thay thế, chưa thay quyết định triển khai cuối cùng.

## Nguồn chính thức

- [Supabase pricing](https://supabase.com/pricing), [pausing](https://supabase.com/docs/guides/platform/free-project-pausing), [function limits](https://supabase.com/docs/guides/functions/limits), [TypeScript runtime](https://supabase.com/docs/guides/functions), [Cron](https://supabase.com/docs/guides/functions/schedule-functions), [Vault](https://supabase.com/docs/guides/database/vault).
- [Streamlit deploy](https://streamlit.io/cloud), [hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [private sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).
- [Appsmith pricing](https://www.appsmith.com/pricing), [REST integration](https://docs.appsmith.com/connect-data/reference/rest-api).
- [Trigger.dev pricing](https://trigger.dev/pricing), [Python extension](https://trigger.dev/changelog/python-extension).
- [Cloudflare Pages pricing](https://developers.cloudflare.com/pages/functions/pricing/), [Workers limits](https://developers.cloudflare.com/workers/platform/limits/).
- [Render Free](https://render.com/docs/free), [Windmill pricing](https://www.windmill.dev/pricing), [Oracle Always Free](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm), [Oracle FAQ](https://www.oracle.com/cloud/free/faq/), [Pipedream pricing](https://pipedream.com/pricing), [HF Spaces](https://huggingface.co/docs/hub/spaces-overview).
