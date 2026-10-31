# Research Portal — đánh giá và hướng triển khai

**BẮT ĐẦU TẠI ĐÂY:** [PoC đơn giản trước](poc-first.md). Khi được yêu cầu build PoC, mặc định Streamlit + Python adapters + refresh thủ công; làm P0–P1 trước. Các stack full platform/cloud bên dưới là lựa chọn mở rộng, không phải yêu cầu cài ngay.

- [Managed/free hosting options](free-managed-options.md): Supabase, UI có sẵn, scheduled Python tasks; không bắt buộc VM.
- [Migration và hosting trên HPC](migration-and-hosting.md): tài liệu/code cần mang sang repo mới, service host, truy cập và storage.
- [Capability audit Kaggle/W&B](provider-capabilities.md): đang chạy/chạy xong, tool hiện có, version gap và backlog adapter.

Đối chiếu checkout ngày **31/10/2026**. Hiện có bộ tài liệu thiết kế và các CLI hỗ trợ; **chưa có ứng dụng portal, API, migration hay frontend triển khai trong checkout**.

Sau PoC profile, đọc [đánh giá mở rộng và quyết định đề xuất](platform-review-2026-10-31.md) khi cần bối cảnh dài hạn. Tài liệu này điều chỉnh những giả định trong bộ thiết kế trước: vị trí repo, dependency isolation, đa tài khoản W&B, worker bền vững, lineage và phạm vi MVP. Đây là đề xuất triển khai, chưa phải tính năng đã làm hoặc quyết định đã được người dùng phê duyệt.

**Phạm vi nghiệp vụ hướng tới, triển khai theo từng increment trong PoC profile:** [Kaggle Operations MVP](kaggle-operations-spec.md) đặc tả Account CRUD UI, notebook status/heartbeat và Submit form ánh xạ flags CLI. Phạm vi này là bản usable đầu tiên theo nhu cầu người dùng, trước viewer/research registry đầy đủ.

Các tài liệu còn lại là nguồn phân tích nghiệp vụ BHSD hữu ích, nhưng không phải đặc tả sẵn sàng triển khai nguyên trạng:

- [Requirements](requirements.md): danh mục nhu cầu nghiên cứu.
- [Architecture](architecture.md): phương án và sơ đồ ban đầu.
- [Data model](data-model.md): thực thể BHSD và schema minh họa, cần sửa theo review.
- [UI/UX](ui-ux.md): ý tưởng màn hình, viewer thuộc giai đoạn sau.
- [Solution landscape](solution-landscape.md): khảo sát ban đầu; xem đính chính trong review.
- [Roadmap](roadmap.md): kế hoạch ban đầu; ưu tiên các mốc nghiệm thu trong review thay lịch cố định.
- [Risks/ADRs](risks-and-decisions.md): các lựa chọn ban đầu, cần cập nhật theo phạm vi platform.
