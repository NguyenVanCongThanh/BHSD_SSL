# Batch size khi huấn luyện

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
