# Audit logic train / resume / validation / evaluation

Ngày: 27/10/2026. Đối chiếu source gốc tại `1bdeb62` và các thay đổi checkpoint đang có trong workspace. Các bản sửa được chia thành các commit theo nhóm loss, core/data, training, evaluation, kiểm thử và tài liệu.

## Kết luận

Có nhiều lỗi ảnh hưởng trực tiếp đến tính đúng đắn; không chỉ là vấn đề tổ chức code. Những lỗi xác nhận dưới đây đã được sửa và có regression test. Việc test qua không chứng minh mọi tổ hợp model/dataset/GPU đều đúng hoặc kết quả cũ vẫn có thể dùng để báo cáo.

## Phát hiện và xử lý

| Mức độ | Vấn đề xác nhận được | Hành vi sau sửa |
|---|---|---|
| Critical | LeFeD backward và optimizer step bên trong `train_step`, rồi engine backward lần nữa trên graph đã dùng | Tách `optimize_batch`; mỗi feedback pass có đúng một backward/update, giữ thứ tự feedback tuần tự và dùng chung AMP/gradient checks |
| Critical | Mẫu unlabeled vẫn dùng nhãn thật trong foreground crop | Ẩn annotation của mẫu ngoài labeled pool trước augmentation; crop và các phép biến đổi sau đó không nhận lesion ground truth của mẫu unlabeled |
| High | Mean Teacher cập nhật EMA trước optimizer; teacher ở eval nên BatchNorm buffers không học | Hook `after_optimizer_step`; EMA cập nhật sau optimizer thành công và sao chép buffers từ student |
| High | CPS mạng thứ hai không có DDP riêng, optimizer không dùng lịch LR chung, checkpoint thiếu mạng/optimizer thứ hai | DDP riêng cho model2; cùng optimizer factory/scheduler; một backward cho tổng loss, hai optimizer do engine quản lý; lưu/restore `net2` và `opt2` |
| High | Dice/consistency có thể cộng lượng lớn voxel bằng FP16, gây overflow/NaN | Probability/reduction dùng FP32; kiểm thử với tensor half lớn đủ vượt giới hạn FP16 |
| High | CE có shared spatial mask chia cho số voxel một mask, không tính số mẫu sau broadcast | Broadcast mask tới toàn bộ target trước khi tính tử và mẫu; loss bất biến khi nhân bản batch |
| High | Tên artifact checkpoint không chứa fold, cloud `latest` có thể trỏ fold khác | Tên mới chứa `_fold{fold}`; kiểm tra metadata fold/task/classes/dataset khi resume/eval |
| High | Seed được đặt trong constructor trainer, sau khi model đã khởi tạo | Seed trước model construction; test hai lần build cùng seed có cùng weights |
| High | Fold-specific split bị thiếu thì âm thầm dùng split mặc định | Khi chỉ định fold, bắt buộc đúng split fold; chặn split rỗng/trùng ID và train/val overlap; benchmark chặn case OOF lặp giữa fold |
| High | `--task` chỉ sửa experiment, trong khi trainer ưu tiên data.task | Đồng bộ task, số lớp, class names ở data/experiment; config file/dataset/model/method không tìm thấy gây lỗi |
| High | `TypeError` bên trong callback bị hiểu là sai số tham số, dẫn đến chạy callback lại | Bind chữ ký trước khi gọi; exception nội bộ truyền ra nguyên vẹn và callback chỉ chạy một lần |
| High | Resume optimizer/scaler lỗi bị warning rồi tiếp tục; resume thiếu latest có thể bắt đầu lại âm thầm | Load state yêu cầu phải khôi phục thành công; thiếu optimizer/latest gây lỗi. PICK cho phép bắt đầu phase self-training mới có chủ đích từ pretrained weights |
| High | Run mới ghi vào thư mục đã có checkpoint, nhưng có thể giữ best từ run cũ | Chặn nếu không bật resume; yêu cầu experiment directory mới |
| High | PICK chuyển phase qua hàm resume nên kéo theo momentum/scaler của pretrain; thiếu pretrained weights vẫn chạy | Transfer chỉ load model strict và tạo lại optimizer; đường dẫn explicit sai hoặc không có pretrained checkpoint gây lỗi |
| High | NaN/Inf có thể làm hỏng weights trước khi bị nhận ra | Kiểm tra scalar loss và gradients trước optimizer; DDP thống nhất quyết định trên mọi rank trước backward/update |
| High | Validation trên DDP dùng BatchNorm buffers khác nhau; best_dice chỉ cập nhật rank 0 | Broadcast buffers trước inference và broadcast best score sau validation; kiểm tra coverage mỗi case đúng một lần |
| High | Sliding window cho phép stride lớn hơn patch tạo voxel không có dự đoán | Validate coverage, shape và probability hữu hạn; softmax/TTA dùng FP32 |
| Medium | Train không đi qua validation cuối thì thiếu best; score 0 không thắng giá trị khởi tạo 0 | Validation thực sự ở cuối nếu chưa trùng lịch; khởi tạo điểm best ở `-inf` để 0 là best hợp lệ. Không tạo best bằng cách gán tên cho final weights |
| Medium | Train/eval tự dựng model bằng hai danh sách kwargs khác nhau và bỏ qua tham số kiến trúc | `core/model_builder.py` dùng chung; checkpoint mới lưu model spec thực tế. Checkpoint cũ dùng đúng tập tham số train cũ từng forward |
| Medium | BCP khai báo `u_weight` nhưng không dùng | Weight voxel labeled=1, pseudo-labeled=`u_weight` trong cả CE và Dice; `u_weight=0` loại gradient loss từ pseudo-labeled voxels |
| Medium | `MaskDiceLoss.class_weights` được lưu nhưng không tham gia loss | Dùng weighted class reduction và validate weights; mặc định weights=1 giữ mean Dice |
| Medium | Extractor chung lấy tensor thứ hai trong multiscale list, có thể lấy scale phụ thay main | Phân biệt PICK `(features, segmentation, features)` với multiscale list main-first |
| Medium | Chỉ số `final_mDice` thực chất là best lịch sử | Ghi riêng `best_mDice` và `last_validation_mDice` |
| Medium | Metrics có thể broadcast prediction/label sai shape; label bị cast uint8 trước kiểm tra | Chặn shape/class-range không hợp lệ trước khi âm thầm tính điểm/cast |

Lỗi tìm checkpoint và các thay đổi schema summary được mô tả riêng trong [BENCHMARK_CHECKPOINT_POLICY.md](BENCHMARK_CHECKPOINT_POLICY.md).

## Các thành phần dùng lại

- `BaseMethod.optimize_batch`, `apply_gradients`, `training_optimizers`, `after_optimizer_step`: một nơi quản lý gradient, AMP và hook sau update. LeFeD mở rộng bằng chuỗi feedback; CPS đăng ký hai optimizer.
- `core/model_builder.py`: phân giải architecture, kwargs thực tế và khởi tạo model; spec được lưu để eval tái dựng chính xác.
- `utils/checkpoint.py`: lưu atomic từng file, manifest theo phase, strict restore và kiểm tra context checkpoint.
- `utils/checkpoint_resolver.py`: thứ tự phase/tên file và chính sách fallback có chủ đích.
- `losses.dice.expand_spatial_mask`: broadcast mask thống nhất cho CE dùng chung và compound loss.

## Kiểm chứng

Môi trường: Python trong `/home/u001015/.venv/nnunet`, PyTorch `2.13.0+cu126`; CUDA không khả dụng trên node này.

```bash
OMP_NUM_THREADS=1 /home/u001015/.venv/nnunet/bin/python -m unittest discover -s tests -q
```

Bộ test cuối: **53 test qua**. Kiểm thử DDP cần Gloo loopback; sandbox chặn loopback nên chạy với quyền đã được cấp. Tất cả artifact kiểm thử nằm trong thư mục tạm.

Test gồm model/loss/optimizer thật, checkpoint serialization thật, resume CPS với momentum, LeFeD feedback, EMA parameters/buffers, NaN, task/config, crop-label isolation, masked-loss scaling, FP16 reductions, sliding-window padding/TTA, và model seed.

Test DDP chạy **hai process CPU/Gloo** cho Mean Teacher, CPS, LeFeD, BCP và PICK với model nhỏ; kiểm tra parameters đồng bộ, hai iteration liên tiếp, rank không có validation case, best score đồng nhất và NaN tại một rank được từ chối trên cả hai rank. Đây chưa phải kiểm thử đa GPU/NCCL hoặc CUDA AMP thực tế.

Test framework cũ đã được cô lập khỏi dataset và experiment thật; không còn lỗi do TensorBoard cố ghi vào thư mục ngoài workspace.

Kiểm tra split thật tại `/home/u001015/notebook/preprocessed_bhsd_pick/splits`:

| Fold | Số train ID | Số val ID | Train/val trùng ID | Val trùng các fold trước |
|---|---:|---:|---:|---:|
| 0 | 2132 | 38 | 0 | 0 |
| 1 | 2131 | 39 | 0 | 0 |
| 2 | 2132 | 38 | 0 | 0 |
| 3 | 2131 | 39 | 0 | 0 |
| 4 | 2132 | 38 | 0 | 0 |

Đây là kiểm tra theo case ID; chưa xác minh quan hệ bệnh nhân/biến thể preprocessing giữa các ID khác tên.

## Tác động đến run cũ và giới hạn còn lại

1. **Run cũ có foreground crop trên ground truth unlabeled không thể được coi là nhãn hoàn toàn ẩn.** Muốn so sánh SSL công bằng cần train lại với chính sách mới và ghi rõ label budget. Sửa eval không khắc phục được leakage trong quá trình train trước đó.
2. **BCP và Dice class weights có thay đổi hành vi khi cấu hình weight khác mặc định tương ứng.** Ghi phiên bản source và policy trong báo cáo; không gộp run cũ/mới như cùng một setting.
3. **CPS legacy checkpoint chỉ có mạng thứ nhất vẫn eval được nhưng không resume đầy đủ được.** Không tự khởi tạo model2 rồi tuyên bố là tiếp tục run cũ.
4. Artifact cloud mới có tên chứa fold; không tự tìm artifact cũ không phân biệt fold. Cloud restore/upload chưa được chạy trong audit này.
5. Resume khôi phục model/optimizer/scaler/iteration; **chưa phải bitwise replay**: chưa serialize RNG theo rank và vị trí iterator/DataLoader. Lịch sample sau resume có thể khác chạy liên tục.
6. PICK hiện dùng unlabeled data qua reconstruction/context, nhưng loss segmentation trong self-training chỉ mask các vùng labeled; chưa có cross-entropy trực tiếp trên pseudo-label vùng unlabeled. Đây cần được đối chiếu với đặc tả thuật toán mong muốn trước khi gọi là tái lập đầy đủ một paper; audit không tự bổ sung một objective nghiên cứu chưa thống nhất.
7. Metric hiện quy ước prediction và ground truth cùng vắng class thì Dice=1. Với lớp hiếm, case mean có thể cao nhờ ca âm tính. Policy được ghi trong summary; cần thống nhất với đối chứng trước khi công bố.
8. Chưa train lại toàn bộ experiment hoặc chạy benchmark inference đầy đủ trên dataset. Chưa chứng minh CUDA AMP, NCCL, throughput, OOM hoặc hội tụ của các model lớn. Thay đổi code cần được kiểm chứng bằng một run GPU ngắn trước các run dài.
9. I/O error trên một rank, hardware failure, filesystem không dùng chung giữa node và lỗi dịch vụ cloud vẫn có thể kết thúc job. Timeout process-group không phải một cơ chế phục hồi job.

Không có cam kết “không còn lỗi”; bằng chứng ở đây xác nhận các lỗi đã tái lập được và phạm vi đã kiểm thử.
