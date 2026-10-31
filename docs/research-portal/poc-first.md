# Điểm bắt đầu triển khai: PoC đơn giản trước

Cập nhật 31/10/2026 theo hướng trao đổi gần nhất. **Đây là implementation profile mặc định khi được yêu cầu xây PoC dựa trên bộ docs này**, trừ khi người dùng yêu cầu stack/phạm vi khác. Không phải phần mềm đã được triển khai hoặc quyền tự submit notebook lên cloud.

## 1. Mục tiêu và stack mặc định

Một người dùng có thể mở UI, chọn account, kiểm tra một notebook Kaggle và đối chiếu một W&B run mà không cần tự gọi CLI.

- UI: **Streamlit chạy trong môi trường local/HPC được phép**.
- Integration: Python adapters tái sử dụng công cụ BHSD qua revision/path cấu hình được.
- Refresh: thủ công bằng nút bấm, ghi fetched_at và lỗi từng nguồn.
- Persistence: account metadata/settings trong file runtime riêng ngoài Git; secret files riêng với quyền truy cập hạn chế. Operation ledger nhỏ bằng SQLite trên local filesystem đã xác minh khi thêm submit. Không đặt SQLite WAL lên shared filesystem chưa kiểm tra locking.
- Chưa yêu cầu Supabase, Trigger.dev, React, FastAPI server, cloud hosting, scheduler, Redis/Celery hay microservices.

Streamlit là server UI của PoC; adapter chứa logic tích hợp, không nhét tất cả vào page callbacks. Giữ ranh giới đủ để thay frontend/storage/runner sau này. Không mặc định deploy Streamlit Community Cloud chỉ vì dùng thư viện Streamlit.

## 2. Thứ tự làm và phạm vi dừng

| Mốc | Kết quả | Nghiệm thu |
|---|---|---|
| P0: khung + fixtures | UI mở được với dữ liệu demo, cấu hình BHSD path/revision | Không cần credential/network/GPU; demo được gắn nhãn rõ |
| P1: read-only thật | Chọn profile, test connection, nhập owner/slug và full W&B run reference, Check now | Hai nguồn có state/time/error riêng; không lẫn identity; thiếu W&B vẫn xem Kaggle được |
| P2: account UI | Thêm/sửa metadata, thay credential, test, disable | Không sửa JSON bằng tay; edit secret trống giữ nguyên; không lộ token qua UI/log/Git |
| P3: build preview | Form flag mapping, build bundle không push | Bundle/task/account/options đúng; preview và bản download đã loại secrets |
| P4: submit có kiểm soát | Submit preview, lưu operation trước dispatch, link sang monitor | Rerun/double click không push trùng; timeout bất định không auto-retry |

**Mặc định hoàn thành P0–P1 trước**, báo kết quả/giới hạn và blockers. P2–P4 là các increment tiếp theo, không cần làm hết mới có PoC dùng được. Nếu người dùng giao đầy đủ account UI/build/submit thì triển khai tiếp theo thứ tự này. Thử read-only dùng credential đã được cấp đúng scope; live push chỉ khi được người dùng yêu cầu/ủy quyền. Không tạo account, thuê service hay nâng cấp plan để hoàn thành PoC mặc định.

## 3. Những điều vẫn phải giữ dù làm nhỏ

- Credential theo account tường minh; không fallback identity máy chủ. Không in cả prefix token.
- Wrapper CLI thay global env phải chạy subprocess cô lập; không gọi trực tiếp trong Streamlit server rồi sửa os.environ chung. Args allowlist, không nhận shell command tùy ý, secret không nằm command args.
- Notebook RUNNING không chứng minh training tiến triển; hiển thị latest iteration/telemetry và timestamps khi có, unavailable khi không có.
- Lỗi network/auth không đổi thành notebook failed hay metric 0. Check thủ công có timeout; không full scan history mỗi lần render/re-run.
- Streamlit rerun không tự gọi push; side effects chỉ theo thao tác rõ ràng và operation state. Session state không thay durable ledger ở P4.
- Không hứa polling khi đóng UI/process. Auto-refresh trong browser, nếu bổ sung, không đồng nghĩa worker thường trực.
- Không đổi training algorithm, không tải dataset/checkpoint lớn để chỉ xem status/summary.

## 4. Cấu trúc repo PoC đề xuất

```text
research-platform/
  app.py
  adapters/
    kaggle.py
    wandb.py
    bhsd.py
  storage/                 # code lưu metadata/secret references; không chứa secret thật
  config/
    settings.example.toml
  tests/
    fixtures/              # sanitized synthetic responses
  docs/
    design/
  requirements.txt
  .gitignore
```

Runtime root ở ngoài checkout, cấu hình bằng env/settings: accounts metadata, private credentials, cache và operation ledger khi cần. Dataset/checkpoints giữ ở BHSD/providers. Cấu trúc apps/services/packages trong review là đích mở rộng; không tạo các service rỗng hoặc cài dependencies chỉ để khớp cây thư mục đó.

## 5. Thứ tự ưu tiên tài liệu

1. Hướng dẫn người dùng hiện hành.
2. **Tài liệu này**: stack và phạm vi mặc định PoC.
3. [Kaggle operations](kaggle-operations-spec.md): quy tắc nghiệp vụ tương ứng increment đang làm; REST endpoints là đề xuất cho full platform, không bắt tạo API server ở PoC.
4. [Provider capabilities](provider-capabilities.md): khả năng, version gap và giới hạn bằng chứng.
5. [Platform review](platform-review-2026-10-31.md), [migration/hosting](migration-and-hosting.md), [managed/free](free-managed-options.md): đích dài hạn và các deployment profile tùy chọn.
6. Architecture/data-model/roadmap/requirements/UI/ADRs ban đầu: tham khảo phần phù hợp, không ghi đè phạm vi PoC.

Khi cần hosting dễ truy cập, persistence nhiều người hoặc scheduled sync khi browser đóng mới chọn profile cloud tương ứng. Không coi mọi nền tảng từng được khảo sát là dependencies bắt buộc.
