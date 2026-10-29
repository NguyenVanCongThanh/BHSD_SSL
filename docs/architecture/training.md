# Kiến trúc huấn luyện

Đối chiếu source ngày 29/10/2026. [Kế hoạch gốc](../archive/plans/base-method-2026-10-28.md) ghi contract trước refactor; các phần tách dưới đây hiện đã có trong repository.

| Thành phần | Trách nhiệm |
|---|---|
| [train.py](../../train.py), [model builder](../../core/model_builder.py) | Resolve config, tạo model spec, dựng model và method |
| [BaseMethod](../../methods/base_method.py) | Vòng đời trainer, phase, training loop, resume và facade gọi helper |
| [Data factory](../../data/dataloader_factory.py) | Dataset, augmentation, loader và sampler |
| [Gradient handler](../../training/gradient_handler.py) | Xử lý tối ưu hóa/gradient dùng chung |
| [Validation runner](../../training/validation_runner.py) | Thực thi evaluation theo options |
| [Validation coordinator](../../training/validation_coordinator.py) | Điều phối validation, chọn best, save, logging và đồng bộ qua callback |
| [Checkpoint metadata](../../training/checkpoint_metadata.py) | Dựng metadata, không serialize hoặc gọi cloud |
| [Checkpoint serializer](../../utils/checkpoint.py) | State, RNG, ghi file và manifest |
| [W&B manager](../../utils/wandb_manager.py) | Run identity, logging và artifact |

Subclass giữ các hook đặc thù thuật toán: Mean Teacher có EMA; CPS có model/optimizer thứ hai; LeFeD có nhiều bước tối ưu; PICK có hai phase. Coordinator dùng callback của trainer để giữ override của subclass khi validate/save, không thay trực tiếp bằng base implementation.

Các contract cần bảo toàn gồm global/per-rank batch, không dùng nhãn unlabeled trong crop, update EMA sau optimizer thành công, thứ tự validation/save, metric policy và phase offset. Các facade/wrapper còn tồn tại không tự được xem là dead code chỉ vì không thấy caller nội bộ.

Các module test tương ứng gồm `test_dataloader_factory`, `test_helpers_step2`, `test_checkpoint_metadata`, `test_validation_coordinator`, `test_base_method_contracts`, cùng tests logic/DDP/AMP. Sự tồn tại của test không chứng minh mọi runtime đã đạt; kết quả kiểm tra phải gắn với môi trường và revision cụ thể.

Xem [training](../guides/training.md), [resume](../guides/checkpoint-resume.md) và [backlog tài liệu/kỹ thuật](../backlog.md).
