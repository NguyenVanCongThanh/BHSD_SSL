# Tài liệu MedSSL-BHSD — Danh mục và Chỉ mục Phân loại

Tài liệu trong thư mục này được phân loại thành **Tài liệu Vận hành (Operational Guides)**, **Kiến trúc & Thiết kế (Architecture Specs)**, và **Hồ sơ Kiểm toán & Phân tích Thực nghiệm (Historical Audits & Empirical Analyses)** để hỗ trợ cả người dùng mới và việc lưu vết nghiên cứu (provenance).

---

## 1. Tài liệu Vận hành & Hướng dẫn sử dụng (Operational Guides)

Các tài liệu này mô tả quy trình thực thi hiện hành, cấu hình và khắc phục sự cố:

* **[KAGGLE_MULTI_ACCOUNT_GUIDE.md](KAGGLE_MULTI_ACCOUNT_GUIDE.md)**: Hướng dẫn tích hợp và điều phối submit notebook tự động trên nhiều tài khoản Kaggle.
* **[KAGGLE_DUAL_GPU_TROUBLESHOOTING.md](KAGGLE_DUAL_GPU_TROUBLESHOOTING.md)**: Sổ tay chẩn đoán và khắc phục lỗi vận hành Dual GPU 2x T4 (DDP, memory, NCCL).
* **[BENCHMARK_CHECKPOINT_POLICY.md](BENCHMARK_CHECKPOINT_POLICY.md)**: Chính sách chuẩn hóa việc lựa chọn checkpoint (`best` vs `latest`) trong đánh giá benchmark.
* **[WANDB_METRICS_AND_DASHBOARD_ORGANIZATION.md](WANDB_METRICS_AND_DASHBOARD_ORGANIZATION.md)**: Quy chuẩn namespace metrics và thiết kế dashboard Weights & Biases (Overview, Evaluation, Diagnostics).
* **[EXPERIMENT_DATA.md](EXPERIMENT_DATA.md)**: Cấu trúc thư mục dữ liệu thực nghiệm ngoại tuyến và quy trình crawl/snapshot.
* **[DDP_BATCH_SIZE.md](DDP_BATCH_SIZE.md)**: Quy ước batch size cục bộ (per-GPU) và toàn cục (global) khi chạy huấn luyện phân tán.

---

## 2. Đặc tả Thiết kế & Kiến trúc (Architecture Specs)

Mô tả chi tiết cấu trúc các module và sự đồng bộ với baseline:

* **[BASE_METHOD_MODULARIZATION_AND_REFACTORING_PLAN.md](BASE_METHOD_MODULARIZATION_AND_REFACTORING_PLAN.md)**: Thiết kế phân tách trách nhiệm trong `BaseMethod` (TrainingLoop, ValidationCoordinator, CheckpointManager, AMP, DDP).
* **[EXPERIMENT_DATA_ARCHITECTURE_AND_AUDIT.md](EXPERIMENT_DATA_ARCHITECTURE_AND_AUDIT.md)**: Chi tiết cấu trúc lưu trữ và phiên bản hóa metadata crawler.
* **[BENCHMARK_ALIGNMENT_MEAN_TEACHER_VS_NNUNET.md](BENCHMARK_ALIGNMENT_MEAN_TEACHER_VS_NNUNET.md)**: Phân tích đồng bộ giữa pipeline huấn luyện Mean Teacher và nnU-Net baseline.
* **[DATASET_COMPARISON_BHSD_FOLD0_VS_NNUNET.md](DATASET_COMPARISON_BHSD_FOLD0_VS_NNUNET.md)**: So sánh phân chia tập dữ liệu Fold 0 và tiền xử lý.

---

## 3. Hồ sơ Kiểm toán & Phân tích Thực nghiệm (Historical Audits & Empirical Analyses)

Hồ sơ lưu trữ các đợt rà soát chất lượng mã nguồn, chẩn đoán lỗi run cụ thể và lịch sử thiết kế:

### Rà soát Mã nguồn & Tinh gọn

* **[AUDIT_TINH_GON_CODEBASE_2026_10_28.md](../AUDIT_TINH_GON_CODEBASE_2026_10_28.md)**: Báo cáo audit tinh gọn codebase ngày 28/10/2026 (trạng thái hiện tại).
* **[IMPLEMENTATION_REVIEW_2026_10_27.md](IMPLEMENTATION_REVIEW_2026_10_27.md)**: Đánh giá chi tiết việc triển khai refactor các phase.
* **[CRITICAL_LOGIC_AUDIT_2026_10_27.md](CRITICAL_LOGIC_AUDIT_2026_10_27.md)**: Kiểm toán các điểm logic then chốt (gradient unscaling, sync DDP).
* **[BENCHMARK_CHECKPOINT_RESOLUTION_AUDIT.md](BENCHMARK_CHECKPOINT_RESOLUTION_AUDIT.md)**: Kiểm toán quy tắc tìm checkpoint benchmark trên 5 fold.
* **[CODEBASE_AUDIT_AND_REFACTOR_PLAN.md](../CODEBASE_AUDIT_AND_REFACTOR_PLAN.md)**: Kế hoạch audit và tái cấu trúc ban đầu.
* **[REFACTORING_PLAN.md](../REFACTORING_PLAN.md)**: Tài liệu lịch sử tái cấu trúc giai đoạn đầu.

### Phân tích Run & Thực nghiệm Cụ thể

* **[RUN_TERMINATION_AUDIT.md](RUN_TERMINATION_AUDIT.md)**: Điều tra nguyên nhân run dừng trước hạn trên Kaggle.
* **[RUN_TERMINATION_AND_MEMORY_ANALYSIS.md](RUN_TERMINATION_AND_MEMORY_ANALYSIS.md)**: Phân tích hiện tượng tràn bộ nhớ GPU và cơ chế giải phóng.
* **[VAL_PREDICTIONS_VISUALIZATION_ANALYSIS.md](VAL_PREDICTIONS_VISUALIZATION_ANALYSIS.md)**: Phân tích ảnh trích xuất validation và mặt cắt dự đoán.
* **[RUN_9TVQ8QLD_INTERIM_ANALYSIS_2026_10_27.md](RUN_9TVQ8QLD_INTERIM_ANALYSIS_2026_10_27.md)**: Báo cáo phân tích run 9tvq8qld từ snapshot offline.
* **[docs/analysis/](analysis)**: Thư mục chứa các artifact báo cáo phân tích tĩnh (bảng dữ liệu, biểu đồ, HTML).

