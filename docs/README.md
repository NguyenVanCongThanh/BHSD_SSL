# Tài liệu MedSSL-BHSD

Bộ hướng dẫn được đối chiếu với source ngày 29/10/2026. Bắt đầu từ [README dự án](../README.md); dùng hướng dẫn theo chủ đề bên dưới. Các báo cáo có ngày là bằng chứng tại snapshot tương ứng, không tự mô tả checkout mới.

## Hướng dẫn hiện hành

| Nhu cầu | Nguồn chuẩn |
|---|---|
| Cấu hình, batch, DDP và ngân sách | [Training](guides/training.md) |
| Lưu checkpoint, local/cloud và policy phục hồi | [Checkpoint/resume](guides/checkpoint-resume.md) |
| Chọn best/latest, strict load và checkpoint legacy | [Checkpoint evaluation](guides/checkpoint-evaluation.md) |
| Điều kiện so sánh SSL, supervised và nnU-Net | [Benchmark protocol](guides/benchmark-protocol.md) |
| Định nghĩa metrics, support, summary và W&B | [Metrics và W&B](guides/metrics-and-wandb.md) |
| Intensity, cohort, panel và error map | [Validation visualization](guides/validation-visualization.md) |
| Crawl, export, snapshot và retention | [Experiment data](guides/experiment-data.md) |
| Bundle, tài khoản và submit | [Kaggle](guides/kaggle.md) |
| Log trống, NCCL và điều tra tài nguyên | [Kaggle troubleshooting](guides/kaggle-troubleshooting.md) |

## Kiến trúc

- [Training và ownership các module](architecture/training.md).
- [Dataset preprocessing và phạm vi bằng chứng đối chiếu](architecture/dataset-preprocessing.md).
- [Việc còn cần kiểm chứng](backlog.md): tách khỏi kết luận lịch sử trong audit.

## Research Management Portal (Thiết kế & Kế hoạch)

- [Tổng hợp yêu cầu & Danh mục chức năng](research-portal/requirements.md)
- [Khảo sát giải pháp & Bảng phân định trách nhiệm](research-portal/solution-landscape.md)
- [Kiến trúc hệ thống, Luồng dữ liệu & Bảo mật](research-portal/architecture.md)
- [Mô hình dữ liệu, Lược đồ ER & JSON Schema](research-portal/data-model.md)
- [Thiết kế UI/UX & Interactive Medical Slice Viewer](research-portal/ui-ux.md)
- [Lộ trình triển khai theo giai đoạn (Roadmap)](research-portal/roadmap.md)
- [Ma trận rủi ro & 10 Quyết định kiến trúc (ADRs)](research-portal/risks-and-decisions.md)

## Phân tích run

- [Run 9tvq8qld tới 16.250 iterations — 28/10](analysis/run-9tvq8qld-16250-2026-10-28.md): train 16.250, validation mới nhất 16.000.
- [Snapshot sớm 3.500 iterations — 27/10](analysis/run-9tvq8qld-3500-2026-10-27.md): giữ để theo dõi diễn biến, không thay thế snapshot sau.
- [Run s9i9nsui: điều tra termination — 26/10](analysis/run-s9i9nsui-termination-2026-10-26.md): NCCL timeout có bằng chứng; nguyên nhân rank ngừng tiến triển còn chưa xác định đầy đủ.

`analysis/` còn chứa CSV/JSON/HTML tái tạo bởi `python tools/analyze_run_9tvq8qld.py`. Script đọc snapshot cố định và ghi các artifact `run_9tvq8qld_*`; không sinh hay ghi đè báo cáo Markdown đã đổi tên. Snapshot thô trong `experiment_data/` là dữ liệu local bị Git ignore.

## Hồ sơ lịch sử

Xem [archive và bảng chuyển đường dẫn](archive/README.md). Audit, kế hoạch và thiết kế cũ được giữ để truy vết; không xuất hiện như hướng dẫn hiện hành. Không lấy test count, inventory, credential setup hoặc checklist trong archive làm trạng thái hiện tại.

Khi cập nhật tài liệu, giữ một nguồn chuẩn cho mỗi contract, dẫn liên kết thay vì lặp định nghĩa. Ghi rõ ngày/revision, phạm vi bằng chứng và phần chưa được kiểm chứng; báo cáo run phải dẫn snapshot bất biến thay vì chỉ alias `latest`.
