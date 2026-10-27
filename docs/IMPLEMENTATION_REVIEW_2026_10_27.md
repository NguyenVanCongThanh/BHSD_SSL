# Rà soát triển khai kế hoạch refactor BHSD_SSL

Ngày kiểm tra: 27/10/2026. HEAD: `b6d4e90` **cộng các thay đổi chưa commit/staged và file mới trong workspace**. Đối chiếu [kế hoạch audit](../CODEBASE_AUDIT_AND_REFACTOR_PLAN.md). Báo cáo này ghi trạng thái sau khi đã xuất hiện các chỉnh sửa mã nguồn; các mục 1–16 trong kế hoạch gốc phản ánh snapshot trước triển khai `8d750d7`.

## Kết luận

Các thay đổi đi đúng hướng nhưng **chưa đạt nghiệm thu và chưa nên dùng cho run dài**. Bộ test hiện xanh chưa bao phủ đường cấu hình thật → model builder. Đã tái hiện lỗi chặn Mean Teacher mặc định, selector dataset bị bỏ qua và preset không được Git đưa vào checkout mới. Ngoài ra còn credential hardcode chưa xử lý, lỗi khôi phục RNG trên CUDA từ đọc mã và lỗi constructor của output adapter với DataParallel.

Lần rà soát này chỉ bổ sung báo cáo/trạng thái tài liệu; không sửa logic triển khai, không thay đổi staging của người dùng, không submit job hay thao tác credential/history.

## Bằng chứng kiểm tra

- `git diff`, staged diff, file mới và kiểm tra ignore; không chỉ so HEAD với tài liệu.
- 108 file Python parse AST thành công; `git diff --check` không báo lỗi.
- Chạy toàn bộ unittest: **89 tests, 88 passed, 1 skipped**, 6.268 giây khi cho phép Gloo loopback. Log lần chạy: `/tmp/bhsd_review_tests_unrestricted.log` (artifact tạm, không phải bằng chứng được version hóa).
- Lần chạy trong sandbox: 1 error do Gloo không resolve loopback; chạy lại ngoài sandbox đã loại trừ lỗi môi trường đó.
- Skip là W&B SDK offline opt-in. Các test mới của AMP dùng GradScaler CPU, không thay CUDA AMP/NCCL integration.
- Chạy probe config/model spec, output/gather, dataset H5 tạm và validator notebook có mock. Không dùng dataset/secret thật trong các probe.
- Quét secret giới hạn trên tracked/untracked source bằng pattern và chỉ báo file/dòng. Chưa quét toàn Git history, mọi loại token hoặc chứng minh token còn hoạt động. Placeholder trong tài liệu được loại khỏi danh sách credential cần xử lý.

## Phần đã làm tốt

| Hạng mục | Bằng chứng hiện tại | Đánh giá |
|---|---|---|
| Dataset YAML không ghi đè hyperparameters Mean Teacher | Probe `load_config(method_config='mean_teacher', dataset_config='bhsd_multiclass')` giữ patch `[32,192,192]`, budget 154 | Đã sửa lỗi ghi đè cũ; vẫn lỗi downstream model spec bên dưới. |
| Xung đột method/model tường minh | Loader fail khi `--config` và selector không khớp | Hướng xử lý hợp lệ theo kế hoạch; cần chuẩn hóa alias và path trước khi so. |
| Aux dictionary bị mất | `extract_multiscale_outputs({'logits': main, 'aux': [aux]})` giữ aux, có test lỗi kiểu/sequence rỗng | Bug cụ thể đã được sửa; extractor chưa có schema thống nhất hoàn chỉnh. |
| AMP nhiều optimizer | Unscale tất cả, finite consensus trước step; test CPU inject inf ở model2 không cập nhật cả hai | Cải thiện đúng, còn thiếu test 2-rank AMP và CUDA. |
| Clipping CPS | `training_models()` trả cả model1/model2, handler clip từng model | Đã xử lý phạm vi model2; cần test clip thực sự và ghi version thay đổi tối ưu. |
| RNG checkpoint | Python/NumPy/Torch lưu bằng tensor/primitives, CPU round-trip với `weights_only=True` qua | Có tiến bộ; chưa đủ cho CUDA resume hoặc RNG theo rank. |
| Fallback dữ liệu | Giữ partition, giới hạn độ sâu, warning; test biên labeled/unlabeled qua | Loại được lỗi crossing partition cụ thể; còn vượt pool thực nghiệm và chưa preflight. |
| Metrics | Có `macro_positive_dice`, eligible classes, support; no-positive trả `None` | Đúng định nghĩa đã đề xuất; chưa log macro W&B hoặc chọn checkpoint riêng. |
| Benchmark | API nhận `folds`, summary có OOF case mean/positive macro | Tiến bộ một phần; chưa xuyên CLI/dataset và thiếu kiểm tra fold IDs. |
| Bảo mật đã sửa | Hai account JSON staged deletion + ignore/template; hai script ban đầu chuyển runtime env; tắt W&B key injection, account list không in prefix | Đúng hướng, nhưng chưa xử lý hết credential source/history/rotation. |
| Tách trainer | Có `GradientHandler`, `ValidationRunner`; giữ hooks và `optimize_batch` | Phù hợp cách tách dần; chưa có `training/engine.py`, không coi Phase 3 hoàn tất. |

## Các lỗi cần sửa trước nghiệm thu

### R01 — P0: vẫn còn credential hardcode ngoài hai script đã sửa

Vị trí xác nhận bằng scan redacted:

- `tools/check_kaggle_error.py:6`: gán token dạng `KGAT_...` vào `os.environ['KAGGLE_KEY']`.
- `tools/check_all_runs.py:5`, `tools/compare_old_new_runs.py:4`, `tools/inspect_wandb_runs.py:5`: gán literal dài cho `WANDB_API_KEY`.

Đây là các file được Git theo dõi và chưa nằm trong diff xử lý credential lần này. Vì vậy chưa thể đánh dấu SEC-01 hoàn tất. Lần audit trước cũng bỏ sót các vị trí này. Không in giá trị hoặc kiểm tra token qua mạng.

**Sửa:** chuyển sang nguồn credential runtime dùng chung, không đưa secret vào argv/log; bổ sung scan tracked tree/notebook outputs ở CI hoặc bước nghiệm thu. Chủ tài khoản rotate/revoke credential và ghi trạng thái history remediation; xóa source không làm key cũ mất hiệu lực. `account add --key` hiện vẫn tồn tại, cần migration riêng.

**Test:** secret scan chỉ báo vị trí, fixture placeholder không bị hiểu là key thật; script thiếu env fail rõ và không gọi mạng ngoài ý muốn.

### R02 — P1: strict model kwargs làm hỏng Mean Teacher mặc định

Vị trí: `core/config_parser.py:417`, `core/model_builder.py:27`, `configs/base.yaml`.

`model.normalization='batchnorm'` kế thừa từ base VNet còn trong cấu hình khi model đổi sang nnUNet3D. Builder mới từ chối field này vì constructor nnUNet3D không nhận `normalization`. Cả cách gọi method/dataset thông thường lẫn preset Mean Teacher đều lỗi **trước khi dựng model/dataset**.

Probe tối thiểu:

```python
import models  # đăng ký registry như train.py
from core.config_parser import load_config
from core.model_builder import model_spec_from_config

cfg = load_config(method_config='mean_teacher', dataset_config='bhsd_multiclass')
model_spec_from_config(cfg, cfg.data.num_classes)
# ValueError: Unsupported model parameter(s) for 'nnunet3d': ['normalization']
```

`load_config(config_path='configs/experiments/bhsd_mt_nnunet_fold0.yaml')` rồi dựng spec cũng lỗi tương tự. Mean Teacher+UNet3D còn giữ cả `anisotropic_kernels/base_filters/deep_supervision/num_stages/use_strided_conv` từ method default. Supervised/CPS/BCP+UNet3D cũng bị `normalization` dư.

**Sửa:** khi model selector thay đổi, thay toàn bộ namespace kiến trúc bằng defaults đúng model rồi áp kwargs của model được chọn; tách VNet defaults khỏi base chung. Giữ strict typo validation, không quay lại bỏ qua mọi field không hỗ trợ. Method defaults của model cũ không được trộn vào model mới.

**Test:** ma trận `load_config → model_spec_from_config → constructor` cho từng preset và các tổ hợp được hỗ trợ. Test hiện chủ yếu kiểm tra config hoặc gọi `MODELS.build()` với kwargs viết tay nên không phát hiện regression này.

### R03 — P1: composition dataset/preset chưa đúng contract

Vị trí: `core/config_parser.py:346`, `core/config_parser.py:402`, `core/config_parser.py:449`.

Ba probe tái hiện:

1. Preset có `dataset: bhsd_binary`, `data: {patch_size: [16,32,32]}`, `method: {name: supervised}` được resolve thành **multiclass 6 lớp**. Khi `data` là dict nhưng không có `name`, nhánh ternary không đọc top-level `dataset`; loader dùng mặc định multiclass.
2. Preset có `data.name: bhsd` bị coi là tên file YAML `bhsd`, trong khi đó là registry name và chỉ có `bhsd_binary.yaml`/`bhsd_multiclass.yaml`. Kết quả `FileNotFoundError`. Round-trip config chứa registry name chưa được hỗ trợ.
3. Custom multiclass 4 lớp khi có `--task multiclass` bị ép về tên/số lớp BHSD 6. Logic vẫn hardcode `current_nc != 6` và fallback `bhsd_multiclass`, trái mục tiêu generic dataset.

Model selector chỉ xuất hiện trong `config_path` cũng chưa nạp YAML model: preset `model: {name: nnunet3d}` + method supervised không nhận `num_stages` từ YAML nnUNet3D. Những preset hiện ghi nhiều field đầy đủ che khuất lỗi composition này.

**Sửa:** xác định một cú pháp selector ổn định tách config identifier và runtime registry name; resolve tất cả selector trước merge, ưu tiên/fail conflict rõ ràng. Không suy số lớp multiclass luôn là 6; metadata từ dataset được chọn phải là nguồn chuẩn. Custom/inline dataset phải có đường load được đặc tả.

**Test:** preset binary với `data` chỉ chứa hyperparameters; model-only defaults; inline/round-trip config; dataset 4 lớp với và không có task override; selector là file path/alias.

### R04 — P1: bốn preset bị ignore và không có trong checkout mới

Vị trí: `.gitignore:6`, `configs/experiments/*.yaml`.

`git check-ignore -v configs/experiments/*.yaml` xác nhận pattern `experiments/` đang match cả `configs/experiments/`. `git ls-files configs/experiments` không có kết quả, và `git status` không liệt kê preset. Các test preset pass nhờ file cục bộ nhưng checkout CI/Kaggle mới sẽ thiếu chúng.

**Sửa:** scope ignore output thành `/experiments/` hoặc thêm exception đúng cho thư mục config, rồi đưa đủ preset vào changeset. Kiểm tra cả các thư mục output ngoài root trước khi thay ignore.

**Test:** kiểm kê manifest file của patch/checkout sạch; mỗi đường preset mà test/document tham chiếu phải được Git track. Không dùng test hiện có trên workspace đầy đủ làm bằng chứng đủ để phân phối preset.

### R05 — P1: CUDA resume mới có nguy cơ lỗi ở khôi phục RNG

Vị trí: `methods/base_method.py:392`, `utils/checkpoint.py:39`, `utils/checkpoint.py:49`.

Trainer đọc checkpoint bằng `map_location=self.device`. Khi device là CUDA, các tensor RNG CPU/NumPy trong checkpoint cũng bị map sang CUDA; `torch.set_rng_state()` cần CPU ByteTensor, còn `keys_tensor.numpy()` không nhận CUDA tensor. CPU RNG round-trip test không bao phủ đường này. Đây là **lỗi từ đọc luồng mã/API cục bộ**, chưa tái hiện CUDA trên node hiện tại.

Ngoài ra checkpoint được lưu bởi rank 0 nhưng tất cả rank restore cùng `rng_state`. Chưa lưu RNG theo rank/world size/sampler epoch/cursor, và training loop vẫn bắt đầu sampler epoch từ 0. Không thể coi REPR-01 hoàn tất hoặc khẳng định replay DDP.

**Sửa:** load state trên CPU hoặc normalize tensor RNG về CPU trước restore; khôi phục CUDA state theo đúng rank/device, kiểm tra topology, không nuốt lỗi restore nếu đang cam kết resume chính xác. Đặc tả checkpoint schema cho rank states/sampler và backward compatibility. Successful-update counter hiện cũng chưa được lưu/restore.

**Test:** CUDA save→load qua entrypoint trainer, RNG NumPy/Torch CPU và GPU; hai rank có RNG khác nhau trước save và vẫn khác/đúng chuỗi sau resume; kiểm tra sampler state theo phạm vi replay đã cam kết.

### R06 — P1 cho API mới: ModelOutput không tương thích DataParallel gather như docstring cam kết

Vị trí: `models/outputs.py:19`.

PyTorch `gather()` dựng dict subclass bằng `type(out)(iterable_of_pairs)`. Constructor `ModelOutput(logits=None, ...)` hiểu positional iterable đó là **logits**, không phải entries của dictionary. Có thể tái hiện lỗi constructor ngay trên CPU, không cần chạy gather tensor CUDA:

```python
from models.outputs import ModelOutput
from torch.nn.parallel.scatter_gather import gather

out = ModelOutput(note=None)
merged = gather([out, out], target_device='cpu')
assert set(merged) == {'note'}  # fail: chỉ có key 'logits', value là generator
```

Hiện backbone cũ chưa đổi sang trả `ModelOutput`, nên lỗi chủ yếu chặn việc sử dụng/migrate API mới; không nói mọi đường train hiện tại đều đã hỏng vì nó.

**Sửa:** constructor hỗ trợ Mapping/iterable theo contract dict hoặc dùng output format được PyTorch hỗ trợ; sửa docstring “guarantees” cho đúng bằng chứng. Adapter LeFeD còn phải biểu diễn hai nhánh/feedback thay vì coi tất cả phần tử sau logits là aux tensor.

**Test:** constructor/copy/serialization, gather CPU của None-valued metadata, và tensor gather trên hai GPU; gradient/autograd cho các nhánh.

### R07 — P1: fallback cùng partition vẫn vượt pool thực nghiệm

Vị trí: `data/dataset_bhsd.py:141`, `methods/base_method.py:setup_dataloaders`.

Sampler giới hạn unlabeled index ở `[labelnum, max_samples)`, nhưng `_get_partition_safe_index()` lấy mọi index ngoài labeled trong **toàn bộ `image_list`**. Probe H5 tạm có 3 case, active pool `[0,1]`, case 1 hỏng và case 2 hợp lệ: `dataset[1]` trả case 2 dù case đó ngoài pool sampler. Có thể vi phạm giới hạn cohort/experimental setting mà không đổi fingerprint.

Giới hạn đệ quy là 5 lần, không phải kiểm kê toàn partition: có thể fail dù còn case tốt xa hơn; file thiếu/HDF5 lỗi chưa có fail policy theo rank. Label `0.5` vẫn được chấp nhận rồi cast thành 0 ở multiclass; binary còn gộp mọi giá trị dương trước khi kiểm tra upper range source label.

**Sửa:** ưu tiên preflight/quarantine theo cohort được xác nhận trước sampler; nếu giữ fallback runtime phải nhận active pool cụ thể, không mở rộng cohort, có provenance/counter và giới hạn theo policy. Validate label nguyên/source mapping trước cast; giữ annotation visibility theo partition.

**Test:** `max_samples < len(dataset)`, toàn pool hỏng nhưng ngoài pool có case tốt, nhiều case liên tiếp hỏng, singleton, missing H5 và fractional/binary out-of-range labels; rank failure test có timeout.

## Phần còn thiếu hoặc cần củng cố

### R08 — P2: ba đường xử lý output chưa thống nhất

- Với `{'metadata': {}, 'custom_logits': tensor}`, multiscale extractor tìm được tensor nhưng `extract_logits()` lỗi vì chỉ fallback value đầu; `wrap_model_output()` trả `logits=None`.
- Với `ModelOutput(logits=main, features=features)` không có aux key, multiscale extractor đưa **features thành aux logits**. Chỉ kiểm tra `isinstance(Tensor)` chưa đủ phân biệt feature channel và class channel.
- Adapter generic còn có thể chấp nhận output không hợp lệ thành `ModelOutput(raw=...)`, trì hoãn lỗi đến consumer.

Cần dùng một chuẩn key/schema chung, không infer features/reconstruction là aux; giữ legacy fallback có chủ đích. Test phải đi từ adapter qua cả extractor và loss, không chỉ từng helper riêng.

### R09 — P2: config validation mới chưa phải schema đầy đủ

`--method.bas_lr 0.02` vẫn được lưu như key mới và bỏ qua, không phát hiện typo. `--experiment.num_classes 3` bị final synchronization đổi lại 6 trước validate, nên xung đột bị che thay vì fail. Các bool có thể qua check `isinstance(x, int)`, và `base_lr=inf` chưa bị chặn. Path selector có thể xung đột với `method.name` chỉ vì chưa canonicalize.

Cần schema/key validation phù hợp extensibility và provenance để phân biệt default cần đồng bộ với explicit value mâu thuẫn; không tự ghi đè explicit input rồi tuyên bố đã kiểm tra invariant.

### R10 — P2: test hiện chưa đủ để nghiệm thu các phase

- Test model shape/trainer thường dựng model bằng kwargs viết tay; test preset chỉ assert section tồn tại. Thêm đường composition thật qua builder để bắt R02/R03.
- Các test Phase 2 chỉ dùng CPU, `is_ddp=False`; tên module `amp_ddp` không đồng nghĩa đã test AMP hai rank. Chưa assert clip norm của cả model, overflow recovery iteration tiếp theo hoặc GradScaler growth policy.
- Golden tests giữ vài loss/metric scalar từ synthetic seed nhưng chưa có metadata source/env tạo baseline, output/gradient/optimizer-state parity cho PICK/Mean Teacher hay inference probability. Đây là smoke numerical regression hữu ích, chưa đủ gate Phase 0/3.
- Fixture metrics có TP=60, FP=20, FN=20 nhưng `dice=0.6`, trong khi Dice từ count là 0.75. Test aggregator hiện lấy field dice nên vẫn pass; nên tạo fixture qua `case_segmentation_stats` hoặc dùng count nhất quán để kiểm tra công thức.
- Chưa có regression cho `ModelOutput`/gather, configurable fold edge cases, offline SDK thật, fresh checkout và local build negative cases.

### R11 — P2: local build mới chỉ kiểm tra JSON/key, có thể báo hợp lệ cho code sai

Vị trí: `tools/kaggle/kaggle_ctl.py:cmd_build`, `NotebookBuilder.build_bundle`.

Probe mock builder trả notebook JSON chứa code `def broken(:` vẫn in `[✓ HỢP LỆ]`. Builder còn catch lỗi generator rồi dùng notebook nguồn/bundle cũ. Vì vậy “build & validate” hiện chưa chứng minh notebook chạy được hoặc vừa sinh đúng revision.

Cần fail khi generator thất bại, validate notebook schema và code cells có hỗ trợ IPython magic/shell, kiểm tra command/config paths, không dùng output cũ im lặng. Thêm test bundle với env key sentinel để xác nhận không nhúng key. Probe lần này không chạy generator thật để tránh ghi đè notebook của người dùng.

### R12 — P2: metrics/folds và tách trainer mới hoàn thành một phần

- Macro/support được ghi vào details/JSON/checkpoint thông qua evaluator, nhưng trainer chỉ log các row per-class và legacy mDice: chưa log scalar `val/macro_positive_dice` lên W&B/TensorBoard, chưa selection policy mới hoặc best-positive file. Giữ legacy default là đúng; không ghi hoàn tất phần tùy chọn này.
- `benchmark_5folds(folds=...)` chưa validate danh sách rỗng/trùng/âm, chưa lưu `requested_folds`; duplicate IDs làm dict collapse và summary không diễn đạt đúng request. `evaluate.py` chưa có cờ folds; BHSD loader vẫn giới hạn 0..4. Chưa đủ Phase 5.
- `dataset_cls` nhận `**kwargs` nhưng không khai báo parameter riêng sẽ không được truyền class metadata qua logic introspection mới; cần contract rõ.
- Có gradient/validation components nhưng loop, phase/checkpoint/logging orchestration còn ở BaseMethod; `training/engine.py` và dataset thứ hai chưa có. Không cần ép thêm engine chỉ để đủ cây thư mục, nhưng phải ghi trạng thái partial.

## Trạng thái nghiệm thu theo kế hoạch

| Phase | Trạng thái sau review | Điều kiện còn thiếu chính |
|---|---|---|
| 0 | **Chưa đạt** | Credential còn trong tracked source; fallback chưa giữ active cohort; rotation/history chưa xác nhận; golden provenance thiếu. |
| 1 | **Chưa đạt** | Mean Teacher build lỗi, preset selector/composition sai, preset bị ignore, unknown-key validation chưa đủ. |
| 2 | **Một phần** | CPU multi-optimizer test qua; CUDA RNG resume/rank state và AMP hai rank chưa kiểm chứng. |
| 3 | **Một phần** | Tách hai component; output API/gather lỗi, chưa parity end-to-end. |
| 4 | **Một phần** | Có macro và build CLI; validation notebook/logging/selection/generator migration chưa đủ. |
| 5 | **Chưa đạt** | Chỉ mở API folds một phần; chưa dataset thứ hai, fold IDs toàn chuỗi hoặc integration tương ứng. |

Thứ tự sửa đề nghị: **R01 credential → R02/R03 composition và R04 preset checkout → R05 CUDA/rank resume → R07 cohort/label validation → R06/R08 output contract → bổ sung test R10 và hoàn thiện R09/R11/R12**. Có thể hoãn sử dụng ModelOutput mới cho đến khi R06 được sửa; không cần đổi mọi forward hiện có trong cùng bản sửa config.

Không đánh dấu phase hoàn tất bằng số file mới, số dòng giảm hoặc tổng số test pass. Gate phải bám đường chạy thực tế, artifact được đưa vào Git và phạm vi CPU/GPU/online đã kiểm chứng.
