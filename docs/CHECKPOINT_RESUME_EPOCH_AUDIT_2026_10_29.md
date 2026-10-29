# Audit checkpoint, resume và ranh giới epoch

Ngày audit: 2026-10-29. Phạm vi: code trong workspace, tập trung Mean Teacher + nnUNet3D và các thành phần checkpoint dùng chung. Tài liệu này tổng hợp, đính chính các trao đổi trước và đề xuất thay đổi; **chưa triển khai thay đổi training**.

## 1. Kết luận

- Training hiện dừng theo tổng iteration. Resume khôi phục checkpoint gần nhất, nhưng tạo iterator DataLoader mới; không tiếp tục chính xác phần còn lại của lượt lấy mẫu đang dở.
- Với 154 ca nhãn và global labeled batch bằng 2, một lượt tập nhãn dài 77 iterations. Điều này phụ thuộc cấu hình thực tế, không phải thuộc tính cố định của hai GPU.
- Resume hiện không lưu permutation/cursor, RNG của từng DataLoader worker hoặc trạng thái prefetch. Không thể bảo đảm chuỗi ca, crop và kết quả giống chạy liên tục.
- `VNet_best_model.pth` là bản lưu tương thích cũ. Khi chạy nnUNet, file này chứa trạng thái nnUNet; tên file không quyết định kiến trúc.
- Checkpoint được ghi local trước. Upload W&B là bước có điều kiện và có lịch riêng; không được coi checkpoint local mới nhất luôn có trên cloud.
- Phương án đề xuất: giữ checkpoint định kỳ, bổ sung checkpoint cuối epoch riêng và chế độ resume tường minh. Ngừng ghi bản best có tên VNet trong lần chạy mới, nhưng tiếp tục đọc checkpoint cũ.

## 2. Bằng chứng trong repo

Các liên kết dưới đây trỏ tới file nguồn; số dòng có thể đổi sau refactor.

| Nội dung | Nguồn |
|---|---|
| Ghép thư mục output, lưu/nạp checkpoint, training loop, lịch upload | [methods/base_method.py](../methods/base_method.py) |
| Lưu và nạp EMA teacher, consistency schedule | [methods/mean_teacher.py](../methods/mean_teacher.py) |
| Permutation, stream nhãn/không nhãn, DDP partition | [data/samplers.py](../data/samplers.py) |
| Dựng DataLoader và augmentation | [data/dataloader_factory.py](../data/dataloader_factory.py) |
| Serialization, RNG, ghi file tạm và manifest | [utils/checkpoint.py](../utils/checkpoint.py) |
| Best/latest sau validation | [training/validation_coordinator.py](../training/validation_coordinator.py) |
| Metadata checkpoint | [training/checkpoint_metadata.py](../training/checkpoint_metadata.py) |
| W&B upload/download, nhận diện run | [utils/wandb_manager.py](../utils/wandb_manager.py) |
| Tìm best checkpoint để evaluation | [utils/checkpoint_resolver.py](../utils/checkpoint_resolver.py) |
| Global/local batch | [core/batching.py](../core/batching.py) |
| Cấu hình method | [configs/methods/mean_teacher.yaml](../configs/methods/mean_teacher.yaml) |
| Preset experiment khác | [configs/experiments/bhsd_mt_nnunet_fold0.yaml](../configs/experiments/bhsd_mt_nnunet_fold0.yaml) |
| Notebook generator Kaggle Mean Teacher | [tools/generate_mean_teacher_notebook.py](../tools/generate_mean_teacher_notebook.py) |

Đây là audit tĩnh. Chưa kiểm tra phiên Kaggle đang chạy, resolved config của phiên đó hay artifact thực có trên tài khoản W&B.

## 3. Đính chính các nhận định trước

| Nhận định | Kết quả audit |
|---|---|
| `max_iteration=30000` | Đúng trong config method và generator Mean Teacher. Preset `bhsd_mt_nnunet_fold0.yaml` lại đặt 12000. Phải xem config sau merge/CLI của phiên chạy. |
| 154 ca nhãn / 2 = 77 iterations/epoch | Đúng khi pool nhãn thực có 154 ca và global labeled batch là 2. DDP hai rank tương ứng mỗi rank 1 ca nhãn/bước. |
| 77 bước mất khoảng 75 giây | Chưa có bằng chứng đo runtime; không suy ra được từ code. |
| 30000 bước tương đương khoảng 390 epochs | Đúng theo nghĩa lượt tập nhãn: 30000/77 ≈ 389,61. Epoch cuối có thể dở. |
| Hơn 2000 ca không nhãn được duyệt hết rồi mới lặp | Không được bảo đảm. Generator không nhãn được tạo lại mỗi epoch tập nhãn; phần chưa dùng của permutation cũ bị bỏ. Số ca còn phụ thuộc pool/max_samples thực tế. |
| Mỗi lần load crop patch 32×192×192 | Đúng cho cấu hình đang xét. Augmentation gồm rotation/flip, scaling và các phép cường độ khi bật extended augmentation; patch size có thể bị override. |
| Checkpoint lưu cả scheduler | Không có scheduler state riêng. LR được tính lại theo iteration và cấu hình; optimizer lưu các param group. |
| Khôi phục RNG đồng nghĩa khôi phục đúng dữ liệu tiếp theo | Sai. Chưa có sampler cursor, worker RNG, trạng thái prefetch; DDP chỉ rank 0 ghi checkpoint. |
| PyTorch không thể resume giữa epoch | Diễn đạt quá tuyệt đối. Repo hiện chưa hỗ trợ; có thể xây dựng sampler/loader có state hoặc phát lại iterator có kiểm soát. |
| Sai lệch luôn <0,2%, hoàn toàn không ảnh hưởng | Không có cơ sở. 77/30000 ≈ 0,257%; tỷ lệ bước cũng không đo được mức ảnh hưởng tới kết quả. Chuỗi các bước sau resume có thể khác toàn bộ. |
| LR và consistency tiếp tục mượt 100% | Bộ đếm schedule được khôi phục nếu giữ nguyên cấu hình. Không bảo đảm loss/gradient/kết quả liên tục vì dữ liệu và nhiễu khác. Consistency còn có nhánh dùng epoch suy ra từ iteration/độ dài loader. |

Không được đồng nhất số iteration với số optimizer update thành công: code có `successful_updates`; AMP có thể bỏ qua update khi gặp gradient không hữu hạn.

## 4. Checkpoint hiện tại lưu gì và khi nào?

### 4.1 Nội dung

Checkpoint của Mean Teacher gồm:

- `net`: student; `ema_net`: teacher EMA.
- `opt`, `scaler` khi có.
- `iter_num`, `best_dice`, `phase`, `checkpoint_kind`.
- RNG Python, NumPy, Torch CPU và CUDA nhìn từ tiến trình lưu.
- Config/metadata: thông tin W&B khi có, metric policy, validation settings, successful updates và một số kết quả validation.

Chưa có: sampler permutation/cursor, số epoch đã hoàn thành để khôi phục vòng lặp, worker RNG, prefetch state, RNG riêng từng DDP rank. Không có LR scheduler state riêng.

### 4.2 Lịch ghi

Với method config Mean Teacher hiện tại:

| Mốc | Hành vi local |
|---|---|
| Mỗi 50 bước, nếu không trùng validation | Ghi đè `checkpoint_latest.pth` |
| Mỗi 250 bước | Validation, sau đó ghi đè latest |
| Validation đạt best mới | Ghi `best_model.pth` và `VNet_best_model.pth` |
| Kết thúc phase | Validation cuối nếu chưa trùng lịch; lưu latest và best nếu đạt |

Hai file best chứa cùng trạng thái mô hình tại cùng iteration; toàn bộ payload không nhất thiết giống từng byte vì được serialize riêng. Best được chọn theo validation; latest dùng để resume, không mặc định là mô hình tốt nhất.

Chỉ main process ghi. Mỗi checkpoint được ghi vào `<file>.tmp`, rồi `os.replace` sang tên chính. Manifest cũng dùng file tạm nhưng cập nhật sau checkpoint: đây không phải một transaction nguyên tử cho cả checkpoint và manifest.

Không có checkpoint khẩn cấp trong nhánh xử lý lỗi của `BaseMethod.train()`. Nhánh đó đóng logging/W&B rồi ném lại lỗi. Cancel/kill không bảo đảm lưu ngay trạng thái đang chạy.

### 4.3 Resume

`--resume` đọc `checkpoint_latest.pth`; nếu local thiếu và W&B đang bật thì thử tải artifact latest. Nếu local đã có, hiện không so sánh iteration với cloud để chọn bản mới hơn.

Sau khi nạp, vòng epoch bắt đầu lại từ 0; DDP gọi `set_epoch(epoch)` với số này. DataLoader tạo lượt lấy mẫu mới. Engine tiếp tục tới tổng iteration đích.

Ví dụ dừng sau bước 18273 nhưng file mới nhất là 18250: khôi phục mô hình ở 18250, còn 11750 bước nếu đích là 30000. Các bước sau 18250 của lần chạy cũ không còn trong trạng thái mô hình được nạp.

## 5. Lưu ở Kaggle hay W&B?

### 5.1 Local

Đường dẫn cho phase đơn:

```text
<experiment.exp_dir>/<experiment.name>/fold_<fold>/
```

Base config đặt `exp_dir=/home/u001015/experiments`. Generator notebook Mean Teacher truyền `EXP_DIR=/kaggle/working/experiments` qua CLI. Vì vậy khi chạy notebook đó, checkpoint local dự kiến nằm tại:

```text
/kaggle/working/experiments/<EXP_NAME>/fold_0/
├── checkpoint_latest.pth
├── best_model.pth
├── VNet_best_model.pth
├── checkpoint_manifest.json
└── log/
```

PICK có thêm thư mục phase `pre_train/`, `self_train/`; không nên dùng đường dẫn phase đơn cho mọi method. File local chỉ là file trên môi trường thực thi; việc giữ lại qua phiên Kaggle mới cần cơ chế xuất/khôi phục output hoặc cloud phù hợp. Không suy ra độ bền lưu trữ từ việc file đã ghi thành công.

### 5.2 W&B

Upload cần `logging.use_wandb=true`, `logging.save_wandb_model=true` và run hoạt động. Config method đang đặt W&B false; notebook/CLI có thể override. Chế độ offline không đồng nghĩa artifact đã có trên cloud.

- Best: gửi artifact khi có best mới, tên dạng `<artifact_prefix>_fold<fold>_best`.
- Latest: chỉ gửi khi `cloud_latest_due()` trả true tại lần save/validation; tên dạng `<artifact_prefix>_fold<fold>_latest`.
- `artifact_interval` mặc định lấy `val_interval`; cuối phase cũng đủ điều kiện upload.
- Lịch được kiểm tra bằng modulo tại các mốc local save/validation. Đặt interval cloud không giao phù hợp với các mốc này có thể khiến upload thưa hơn dự kiến.
- `run.log_artifact()` được gọi để đưa upload vào hàng đợi. Log “Queued” không chứng minh upload đã hoàn tất khi tiến trình bị kill.
- File legacy best không được upload thành artifact riêng qua nhánh legacy; artifact best chứa file canonical.

Ví dụ local lưu mỗi 50 và cloud mỗi 250: local có thể ở bước 18200 trong khi cloud mới ở 18000. Không thể resume từ một file chỉ có ở phiên Kaggle cũ nếu phiên mới chưa khôi phục file đó.

## 6. Phương án 1 đề xuất: checkpoint cuối epoch riêng

### 6.1 Mục tiêu và giới hạn

Mục tiêu: khi chọn resume theo epoch, bắt đầu từ ranh giới đã duyệt xong tập nhãn, không nối phần còn lại của epoch dở bằng permutation mới.

Không đặt mục tiêu tái hiện bitwise, giữ crop giống hệt, hoặc bảo đảm duyệt hết pool không nhãn. Khi kích thước pool không chia hết batch, sampler có drop remainder; “hết epoch” chỉ nghĩa là hết các batch sampler phát ra, không luôn là mọi ca đều xuất hiện.

### 6.2 Hành vi mong muốn

Giữ lịch save/validation hiện tại và bổ sung:

```text
checkpoint_latest.pth         # Phục hồi gần thời điểm dừng
checkpoint_epoch_latest.pth   # Phục hồi tại cuối lượt tập nhãn
best_model.pth                # Mô hình được chọn theo validation
```

Chế độ cấu hình đề xuất, chưa tồn tại trong code:

```yaml
experiment:
  resume: true
  resume_policy: epoch_boundary  # hoặc latest; mặc định latest để tương thích
```

- Tính ranh giới từ iterator thực tế/`len(trainloader)`, không hardcode 77.
- Lưu sau khi hoàn tất batch cuối; DDP cần các rank thống nhất điểm lưu.
- Nếu validation trùng ranh giới, thống nhất thứ tự validation/save để metadata best và RNG không mâu thuẫn do thứ tự callback.
- Thêm metadata tối thiểu: schema version, `completed_epochs`, `epoch_length`, `at_epoch_boundary`, batch layout, kích thước/định danh split.
- Resume khôi phục epoch tiếp theo từ `completed_epochs`, không từ 0; kiểm tra cấu hình/split có tương thích.
- Cloud có artifact riêng `<prefix>_epoch_latest`, chứa đúng file cuối epoch. Chọn policy nào thì tải đúng artifact đó, không lấy artifact latest giữa epoch thay thế âm thầm.
- Có thể upload mỗi N epoch để giảm chi phí. Cần ghi rõ cloud phục hồi được tới epoch nào; không dùng nguyên modulo iteration có thể không giao lịch epoch.
- Metadata `checkpoint_kind` cần loại tường minh cho checkpoint epoch; hiện tên mới sẽ bị đánh dấu `unverified` nếu không sửa.
- `max_iteration` không chia hết epoch vẫn được giữ. Ví dụ 30000 = 389×77 + 47: latest cuối cùng là 30000, checkpoint cuối epoch gần nhất là 29953. Không đánh dấu bước 30000 là cuối epoch.

Ví dụ: checkpoint epoch ở 154, latest ở 200, dừng sau 220. Policy epoch quay lại 154, policy latest quay lại 200. Resume theo epoch đánh đổi nhiều công việc chạy lại hơn để giữ ranh giới lượt nhãn.

### 6.3 Trường hợp cần quyết định rõ khi triển khai

- Trước epoch đầu chưa có checkpoint epoch: báo thiếu checkpoint hoặc khởi chạy mới theo lựa chọn tường minh, không giả vờ resume thành công.
- Checkpoint cũ giữa epoch: không thể chuyển thành cuối epoch bằng sửa `iter_num`. Có thể nạp theo policy latest, chạy trọn iterator mới rồi lưu ranh giới mới. Khi đó phải theo dõi số batch thực của lượt mới; không dùng đơn thuần `iter_num % 77`.
- Checkpoint cũ đúng bội số 77 nhưng thiếu metadata: không tự khẳng định là boundary nếu phiên đó từng resume giữa epoch; cần xác minh lịch sử hoặc coi là legacy không xác định.
- Thay đổi world size, batch size hoặc pool làm epoch length đổi: không suy luận epoch bằng phép chia mới; báo không tương thích hoặc chuyển mode tường minh.
- Với PICK hai phase: epoch counter, artifact prefix và checkpoint phải tách theo phase.
- Khi quay lùi checkpoint, W&B/TensorBoard có thể đã có log iteration cao hơn: ghi rõ resume source và xử lý step/run identity để không âm thầm mất log. Không tự xóa best mới hơn đã tồn tại; phân biệt best của lịch sử run với trạng thái best được nạp.
- Epoch boundary chỉ tránh nối epoch dở; DDP vẫn chưa có RNG từng rank và worker. Không quảng bá là exact resume.

## 7. Sửa tên VNet như thế nào?

Đề xuất ngừng ghi mới `VNet_best_model.pth`, giữ canonical `best_model.pth`. Không đổi thành tên kiến trúc cụ thể.

Các nơi cần sửa/kiểm tra cùng nhau:

1. `BaseMethod.run_training_loop`: mặc định `legacy_best_model_name=None`; cập nhật lời gọi PICK còn truyền tên legacy.
2. `validation_coordinator`: bỏ nhu cầu ghi bản sao legacy trong flow chuẩn.
3. Giữ fallback đọc legacy trong `checkpoint_resolver` và logic chọn pretrain cũ của PICK.
4. Kiểm tra fresh-run guard/nhận diện W&B: vẫn nhận thư mục có checkpoint legacy để tránh ghi đè kết quả cũ.
5. Sửa generator notebook Kaggle còn hướng dẫn/lựa chọn tên VNet; regenerate notebook tương ứng nếu notebook được quản lý trong repo.
6. Giữ test đọc checkpoint VNet cũ; thay test đòi ghi hai file best bằng test chỉ ghi canonical.

Không xóa/đổi tên hàng loạt checkpoint cũ. Bỏ bản sao mới giảm ghi đĩa và dung lượng, không làm đổi kiến trúc hay trọng số.

## 8. Danh sách triển khai và mức ưu tiên

| Ưu tiên | Công việc | Thành phần |
|---|---|---|
| P1 | Chỉ ghi best canonical, vẫn đọc legacy | BaseMethod, PICK, validation, resolver, notebook |
| P1 | Thêm checkpoint epoch và metadata boundary | Training loop, checkpoint metadata/serialization |
| P1 | Thêm policy chọn nguồn resume, restore epoch và kiểm tra tương thích | Config parser, CLI/notebook, BaseMethod |
| P1 | Upload/download epoch artifact riêng, kiểm tra thiếu file sau download | W&B manager và engine |
| P1 | Nhận diện file mới trong manifest, fresh-run guard và run identity | Checkpoint, W&B manager |
| P2 | Log rõ local path, checkpoint iteration, epoch, policy, artifact được queue/khôi phục | Engine/W&B |
| P2 | Hướng dẫn khôi phục Kaggle từ local output/cloud và cấu hình đã resolve | Notebook/docs |
| P3 | Dừng có kiểm soát sau batch rồi lưu | Engine/DDP; không bảo đảm với SIGKILL |
| P3 | Resume chính xác giữa epoch bằng sampler state và augmentation seed | Sampler/DataLoader; thay đổi lớn hơn phương án 1 |

Không chỉ đổi `save_interval=77`: validation vẫn ghi latest giữa epoch, DDP vẫn reset epoch, cloud chưa có policy tương ứng và epoch length có thể đổi.

## 9. Tiêu chí kiểm chứng khi thực hiện sửa code

- Loader giả có epoch ngắn: latest vẫn theo lịch bước, epoch checkpoint chỉ được ghi sau batch cuối.
- Resume checkpoint epoch: batch đầu thuộc epoch kế tiếp; `set_epoch()` nhận số khôi phục đúng trên các rank.
- Validation giữa epoch không ghi đè epoch checkpoint; khi trùng boundary, metadata nhất quán.
- Dừng giữa epoch và resume hai policy chọn đúng nguồn và tổng iteration đích.
- Dừng cuối training giữa epoch không tạo boundary giả; ví dụ 30000/77 ở trên.
- Metadata thiếu/không tương thích, chưa có epoch checkpoint và legacy migration có hành vi rõ ràng.
- W&B mock kiểm tra tên artifact, file tải về, điều kiện upload; không upload thực trong unit test.
- Best mới chỉ có canonical; checkpoint legacy vẫn đọc/evaluate được.
- Fresh-run guard không bỏ sót thư mục chỉ có checkpoint epoch; phase PICK không lẫn nhau.
- Smoke test DDP hai rank kiểm tra không deadlock và số batch đồng nhất; không cần chạy đủ 30000 bước.

Trong lần audit tài liệu này chưa chạy training hoặc test runtime vì không sửa code thực thi. Các phép tính và nhánh xử lý được đối chiếu trực tiếp với mã nguồn.
