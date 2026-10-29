# Archive và chuyển đường dẫn

Đợt rà soát tài liệu ngày 29/10/2026 giữ hồ sơ lịch sử để bảo toàn provenance. Các heading, số liệu, lệnh và kết luận trong bản gốc phản ánh thời điểm cũ; banner đầu file chỉ tới nguồn thay thế khi cần. Không thực thi lệnh cũ mà chưa đối chiếu hướng dẫn hiện hành.

[Chỉ mục hiện hành](../README.md) · [Backlog](../backlog.md)

## Bảng chuyển đường dẫn

Các tên cũ dưới đây đã được thay bằng đường dẫn mới, không giữ stub trùng nội dung. Liên kết ngoài repository tới tên cũ cần cập nhật.

| Đường dẫn cũ | Vị trí mới |
|---|---|
| `REFACTORING_PLAN.md` | [refactoring-2026-10-15.md](plans/refactoring-2026-10-15.md) |
| `CODEBASE_AUDIT_AND_REFACTOR_PLAN.md` | [codebase-refactor-2026-10-27.md](audits/codebase-refactor-2026-10-27.md) |
| `AUDIT_TINH_GON_CODEBASE_2026_10_28.md` | [codebase-simplification-2026-10-28.md](audits/codebase-simplification-2026-10-28.md) |
| `docs/IMPLEMENTATION_REVIEW_2026_10_27.md` | [implementation-review-2026-10-27.md](audits/implementation-review-2026-10-27.md) |
| `docs/CRITICAL_LOGIC_AUDIT_2026_10_27.md` | [critical-logic-2026-10-27.md](audits/critical-logic-2026-10-27.md) |
| `docs/BENCHMARK_CHECKPOINT_RESOLUTION_AUDIT.md` | [checkpoint-resolution-2026-10-27.md](audits/checkpoint-resolution-2026-10-27.md) |
| `docs/CHECKPOINT_RESUME_EPOCH_AUDIT_2026_10_29.md` | [checkpoint-resume-2026-10-29.md](audits/checkpoint-resume-2026-10-29.md) |
| `docs/BASE_METHOD_MODULARIZATION_AND_REFACTORING_PLAN.md` | [base-method-2026-10-28.md](plans/base-method-2026-10-28.md) |
| `docs/EXPERIMENT_DATA_ARCHITECTURE_AND_AUDIT.md` | [experiment-data-2026-10-28.md](audits/experiment-data-2026-10-28.md) |
| `docs/BENCHMARK_SCALING_AND_TRAINING_EFFICIENCY_GUIDE.md` | [benchmark-scaling-2026-10-29-snapshot.md](plans/benchmark-scaling-2026-10-29-snapshot.md) |
| `docs/RUN_TERMINATION_AND_MEMORY_ANALYSIS.md` | [run-s9i9nsui-memory-2026-10-25-superseded.md](audits/run-s9i9nsui-memory-2026-10-25-superseded.md) |
| `docs/RUN_9TVQ8QLD_INTERIM_ANALYSIS_2026_10_27.md` | [run-9tvq8qld-3500-2026-10-27.md](../analysis/run-9tvq8qld-3500-2026-10-27.md) |
| `docs/analysis/RUN_9TVQ8QLD_16250_ANALYSIS_VI.md` | [run-9tvq8qld-16250-2026-10-28.md](../analysis/run-9tvq8qld-16250-2026-10-28.md) |
| `docs/RUN_TERMINATION_AUDIT.md` | [run-s9i9nsui-termination-2026-10-26.md](../analysis/run-s9i9nsui-termination-2026-10-26.md) |
| `docs/BENCHMARK_CHECKPOINT_POLICY.md` | [checkpoint-evaluation.md](../guides/checkpoint-evaluation.md) |
| `docs/KAGGLE_MULTI_ACCOUNT_GUIDE.md` | [kaggle.md](../guides/kaggle.md) |
| `docs/KAGGLE_DUAL_GPU_TROUBLESHOOTING.md` | [kaggle-troubleshooting.md](../guides/kaggle-troubleshooting.md) |
| `docs/DATASET_COMPARISON_BHSD_FOLD0_VS_NNUNET.md` | [dataset-preprocessing.md](../architecture/dataset-preprocessing.md) |
| `docs/VAL_PREDICTIONS_VISUALIZATION_ANALYSIS.md` | [validation-visualization.md](../guides/validation-visualization.md) |
| `docs/WANDB_METRICS_AND_DASHBOARD_ORGANIZATION.md` | [metrics-and-wandb.md](../guides/metrics-and-wandb.md) |
| `docs/BENCHMARK_ALIGNMENT_MEAN_TEACHER_VS_NNUNET.md` | [benchmark-protocol.md](../guides/benchmark-protocol.md) |
| `docs/EXPERIMENT_DATA.md` | [experiment-data.md](../guides/experiment-data.md) |
| `docs/DDP_BATCH_SIZE.md` | [training.md](../guides/training.md) |

## Bản thiết kế giữ thêm trước khi rút gọn

- [Benchmark alignment](plans/benchmark-alignment-before-doc-review-2026-10-29.md).
- [W&B metrics/dashboard](plans/wandb-design-before-doc-review-2026-10-29.md).

Ngày trong tên hai bản trên là ngày lưu snapshot tài liệu, không suy thành ngày phát sinh mọi phát hiện. Bản scaling cũng dùng ngày lưu snapshot. Phần hướng dẫn experiment data đã gộp; inventory cũ vẫn nằm trong audits.
