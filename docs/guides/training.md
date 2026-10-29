# Huấn luyện và cấu hình

Đối chiếu source ngày 29/10/2026. Entry point: [train.py](../../train.py); parser: [core/config_parser.py](../../core/config_parser.py).

## Chọn cấu hình

```bash
python train.py --dataset bhsd_multiclass --method mean_teacher --fold 0 \
  --data.root_path /path/to/BHSD_fold0 --experiment.name mt_fold0

torchrun --standalone --nproc_per_node=2 train.py \
  --dataset bhsd_multiclass --method mean_teacher --fold 0 \
  --data.root_path /path/to/BHSD_fold0 --experiment.name mt_ddp_fold0
```

Config thường được ghép theo base → dataset → method → experiment preset → CLI; model được dựng từ cấu hình kiến trúc riêng để tránh lẫn tham số model khác. Selector mâu thuẫn với preset có thể gây lỗi. Dùng `--config configs/experiments/...yaml` khi cần preset hoàn chỉnh; kiểm tra `resolved_config.json` của run thay vì suy cấu hình từ tên notebook.

Ngân sách Mean Teacher nằm ở `method.max_iteration`, không phải `experiment.max_iterations`. Method default đặt 30000; preset `bhsd_mt_nnunet_fold0.yaml` đặt 12000. LR, consistency, validation và save cần được xem cùng cấu hình đã resolve.

## Batch size và DDP

`data.batch_size` và `data.labeled_bs` luôn là số mẫu **tổng mỗi bước**,
trên tất cả GPU. Config gốc được giữ nguyên khi ghi W&B.

Với Mean Teacher, CPS hoặc LeFeD:

| Thiết lập | Tổng | Mỗi GPU khi DDP 2 GPU |
|---|---:|---:|
| batch_size | 4 | 2 |
| labeled_bs | 2 | 1 |
| unlabeled | 2 | 1 |

Trainer dùng số labeled cục bộ để tính supervised loss. Một tiến trình
(CPU, single GPU hoặc DataParallel) giữ nguyên batch tổng. Trong DataParallel,
PyTorch chia input ở forward, không chia batch tại DataLoader.

DDP yêu cầu cả labeled và unlabeled chia hết cho world_size. Cấu hình
không hợp lệ báo lỗi trước khi tạo model DDP; không tự làm tròn.
PICK và BCP hiện cần số labeled/unlabeled bằng nhau, chẵn và ít nhất 2 mỗi
rank do ghép cặp CutMix. Với hai GPU, dùng tổng batch_size=8, labeled_bs=4
cho các phương pháp này. Không áp dụng giới hạn đó cho Mean Teacher.

Supervised lấy batch hoàn toàn từ pool labeled (`labelnum`), dùng đầy đủ
batch_size; labeled_bs cấu hình không khiến trainer bỏ bớt mẫu trong batch.
Batch cuối không đủ kích thước được bỏ. Sampler SSL DDP bỏ phần dư của
pool khi chia rank để hai rank luôn có cùng số bước mỗi epoch.

Khi khởi động, log in global batch, world_size và batch mỗi rank; W&B summary
lưu thêm `batch_layout`. Có thể giữ notebook Mean Teacher ở BATCH_SIZE=4,
LABELED_BS=2. Các run DDP trước bản sửa có thể đã dùng global batch lớn hơn
world_size lần; cần phân biệt khi so sánh hoặc resume thí nghiệm cũ.

Kiểm tra: `python -m unittest tests.test_batching -v`. Các test tích hợp sampler
cần môi trường PyTorch cùng dependencies của project. Smoke test trên hai GPU
cần xác nhận log `global=4 (2L+2U), world_size=2, per_rank=2 (1L+1U)` và chạy
qua backward, validation, lưu/load checkpoint; không thể thay thế bằng test
phép chia batch trên CPU.


## Epoch và tối ưu thời gian

Với sampler SSL, độ dài lượt lấy mẫu phụ thuộc pool labeled và labeled batch, có bỏ phần dư khi chia batch/rank. Dùng `len(trainloader)` thực tế; không lấy tổng số ảnh chia global batch để suy epoch cho mọi method. Ví dụ 154 labeled và global labeled batch 2 cho 77 bước khi partition thực tế phù hợp.

Tăng batch làm đổi exposure và số optimizer updates; giảm số bước theo tỷ lệ batch không bảo đảm giữ nguyên hội tụ. Profile riêng train, validation, render, upload và checkpoint trước khi chỉnh. Không suy VRAM đỉnh hoặc throughput từ một ảnh chụp mức dùng bộ nhớ.

Giảm tần suất validation có thể bỏ lỡ best; đổi stride/TTA làm đổi protocol inference. Ghi cấu hình và đo lại trước khi so sánh kết quả. Khi thay ngân sách, kiểm tra lại LR/consistency schedule và nhu cầu checkpoint phục hồi.

Xem [checkpoint và resume](checkpoint-resume.md), [benchmark](benchmark-protocol.md), [Kaggle](kaggle.md).
