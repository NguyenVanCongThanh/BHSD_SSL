# Kế hoạch module hóa và refactor `BaseMethod`

Trạng thái: đặc tả triển khai, chưa thực hiện refactor. Đối chiếu mã nguồn ngày 2026-10-28.

## 1. Mục tiêu và phạm vi

[`methods/base_method.py`](../methods/base_method.py) hiện có 907 dòng và 26 phương thức. Lớp vừa điều phối training, vừa dựng data pipeline, cấu hình runtime, đóng gói checkpoint, xử lý validation và xuất log. Những phần có lý do thay đổi khác nhau nên được tách ra để dễ kiểm thử và bảo trì.

Mục tiêu của đợt này là **giảm phụ thuộc và làm rõ luồng điều phối trong khi bảo toàn hành vi hiện tại**. Số dòng chỉ là chỉ số tham khảo; không đặt yêu cầu rút xuống 250–280 dòng hoặc ép loop còn 60–80 dòng.

Trong phạm vi:

- Tách data factory với kết quả trả về có tên trường.
- Tách phần dựng bảng W&B, thu thập thông tin môi trường và định dạng tiến độ.
- Tách đóng gói metadata checkpoint khỏi serializer.
- Tách điều phối validation định kỳ, giữ inference trong `ValidationRunner`.
- Giữ các phương thức hiện hữu của `BaseMethod` làm facade ủy quyền.
- Bổ sung kiểm thử tại các ranh giới có nguy cơ hồi quy.

Ngoài phạm vi: thay đổi loss, augmentation, sampler, learning-rate schedule, AMP/DDP policy, định nghĩa metrics, định dạng checkpoint, resume semantics hoặc thuật toán các method. Không cache callback signature, đổi cơ chế DDP hay tái thiết kế toàn bộ runtime trong đợt này.

## 2. Hiện trạng đã kiểm chứng

| Thành phần | Hiện trạng | Hướng xử lý |
| --- | --- | --- |
| `setup_dataloaders()` | Dựng transforms, dataset qua registry, phân vùng nhãn và sampler | Chuyển sang data factory |
| `evaluate_validation_model()` | Tạo `ValidationRunner` mới mỗi lần, gán `validation_details` | Giữ cơ chế tạo mới và delegation |
| `ValidationRunner.run()` | Chuyển tiếp inference options, visual cohort/settings, nhận details và gắn `weights_source` | Giữ chuyên trách đánh giá; không gọi là “chỉ dùng 30% công năng” |
| `_handle_periodic_validation()` | Gọi hook validation, chọn best, lưu checkpoint, xuất log/JSON và DDP sync | Chuyển orchestration sang coordinator |
| `save_checkpoint()` | Tạo metadata, gọi serializer, rồi upload artifact | Tách metadata; giữ hook và thứ tự side effect |
| `load_checkpoint()` | Kiểm tra context, khôi phục model/optimizer/scaler/RNG và counters | Giữ tại facade trong đợt đầu |
| `run_training_loop()` | Điều phối phase, resume/cloud restore, sampler epoch, train/log/save/validate và đóng session | Giữ quyền sở hữu vòng đời; chỉ tách tiếp khi có nhu cầu cụ thể |
| Visual wrappers | Loop hiện dùng `build_visual_payload()`; hai wrapper còn lại chưa có caller nội bộ ngoài định nghĩa | Giữ cả ba để bảo toàn API |
| `train()` | Có nhánh cleanup khi lỗi nhưng dùng `sys` chưa import ở cấp module | Sửa lỗi riêng, có kiểm thử và commit riêng |

Không coi việc không tìm thấy caller nội bộ là bằng chứng đủ để xóa public method. Các cam kết tương thích phải được kiểm chứng bằng contract tests, không suy ra chỉ từ việc giữ nguyên chữ ký.

## 3. Hợp đồng bắt buộc bảo toàn

### 3.1. Hook, trạng thái và vòng đời

- Giữ tên, chữ ký, giá trị mặc định và giá trị trả về của toàn bộ phương thức `BaseMethod`, kể cả các wrapper trực quan.
- Giữ thứ tự khởi tạo: environment → data → optimizer → logging; không gọi hook cần EMA/model2 trước khi subclass khởi tạo xong.
- Coordinator gọi **`self.validate` và `self.save_checkpoint` thông qua callback được truyền vào**. Không thay bằng lời gọi trực tiếp base implementation hoặc low-level serializer.
- Giữ `validation_details`, `last_validation_dice`, `successful_updates`, `_phase_max_iters` và các thuộc tính dataset/loader/sampler mà loop hoặc subclass sử dụng.
- Không sửa tests bằng cách bỏ assertion hành vi. Có thể cập nhật mock target khi ownership của import đổi, với lý do rõ ràng.

| Method | Hợp đồng đặc thù |
| --- | --- |
| PICK | Hai phase dùng chung engine; giữ `start_step_offset`, `allow_new_phase`, artifact prefix, snapshot riêng, chuyển pretrain weights và thời điểm đóng writer/W&B |
| Mean Teacher | Validation qua EMA, `weights_source='ema_net'`; checkpoint giữ `ema_net`; EMA chỉ cập nhật sau optimizer step thành công |
| CPS | Giữ cả hai optimizer/model, LR của optimizer2 và checkpoint `net2`/`opt2` |
| LeFeD | Giữ override `optimize_batch()` và số lần backward/update của recurrent feedback |
| BCP / Supervised | Giữ batch partition và labeled-only data path của supervised |

### 3.2. Data và ngẫu nhiên

Giữ nguyên dataset registry, dataset name, thứ tự và tham số transforms, split/fold, overlap check, `labelnum`/`max_samples`, intensity metadata và normalization. Gán labeled/unlabeled indices trước khi có augmentation phụ thuộc nhãn.

Giữ labeled-first ordering, kích thước global/per-rank, kiểm tra đủ pool cho một global batch, supervised labeled subset, seed sampler, `set_epoch()`, `drop_last`, `num_workers`, `pin_memory` và shuffle policy. Không phát sinh thêm random draw hoặc đổi thời điểm khởi tạo thành phần tiêu thụ RNG.

### 3.3. Validation, logging và checkpoint

- Validation chạy trên mọi rank theo luồng collective hiện tại, kể cả rank không có validation case. Chỉ rank chính xuất file, log và upload artifact.
- Kiểm tra nonfinite metrics trước khi chọn best. Dice bằng 0 là best hợp lệ khi best ban đầu là `-inf`; chỉ cải thiện nghiêm ngặt (`>`) mới thay best.
- Giữ cập nhật iteration/global step/phase vào details trước khi lưu checkpoint.
- Thứ tự hiện tại: best và legacy nếu cải thiện → latest → validation payload/JSON → broadcast best/barrier → `self.model.train()`.
- Gom metrics, ảnh và bảng trong **cùng một validation payload** gửi qua một lần `log_metrics()`. Đây không phải bảo đảm giao dịch giữa W&B, file và checkpoint.
- Giữ tên metric, 16 cột bảng và mapping field hiện tại, foreground class names, visual cohort/cadence, `global_step` và W&B summary.
- JSON validation tiếp tục dùng file tạm rồi replace, `allow_nan=False` và cùng schema.
- Không tạo best từ final weights chưa được validation; vẫn validation cuối phase nếu iteration cuối không trùng chu kỳ.
- Giữ schema checkpoint/manifest, atomic replace, context validation, strict loading, scaler/RNG state và config redaction. `extra_state` của caller tiếp tục ghi đè metadata theo merge order hiện tại.
- Giữ thứ tự lưu local thành công trước upload; giữ artifact name/type/aliases và `cloud_latest_due()`.

### 3.4. Resume và kết thúc phase

Giữ cloud restore khi thiếu latest local, barrier trước load, từ chối ghi đè khi chưa bật resume, lỗi missing latest, nhánh `allow_new_phase`, skip phase đã hoàn thành và phase/global-step offset.

Không tuyên bố resume tái tạo chính xác trajectory nếu chưa kiểm chứng sampler/worker/RNG ở mức tương ứng. Đợt này bảo toàn semantics hiện có, không mở rộng cam kết reproducibility.

## 4. Kiến trúc đích và giao diện

```text
methods/base_method.py                # Facade, runtime setup, hooks, training/phase lifecycle
  ├── data/dataloader_factory.py      # Dataset/transforms/loader/sampler
  ├── training/validation_runner.py   # Inference, metrics, visual records (hiện có)
  ├── training/validation_coordinator.py # Periodic validation orchestration
  ├── training/checkpoint_metadata.py # Metadata thuần dữ liệu
  └── utils/
      ├── system_info.py             # Git/runtime info và resolved config
      ├── logger.py                  # Progress formatting
      ├── wandb_manager.py           # SDK table building và logging
      └── checkpoint.py              # Serializer/manifest/RNG (hiện có)
```

Helper/coordinator không import `methods.base_method` và không nhận toàn bộ trainer làm context. `BaseMethod` nối các thành phần bằng dữ liệu và callback. Không tạo `CheckpointCoordinator` chỉ để bọc lại serializer; artifact và resume orchestration vẫn ở facade/loop trong đợt này.

### 4.1. Data factory

Dùng `BatchLayout` hiện có của `core/batching.py`, `Config` và kiểu dataset/sampler phù hợp thay cho tuple nhiều `Any`. Tại biên dataset registry có thể cần kiểu linh hoạt, nhưng phải ghi rõ thuộc tính được sử dụng.

```python
@dataclass
class DataLoaderBundle:
    db_train: Dataset
    db_val: Dataset
    trainloader: DataLoader
    train_sampler: Optional[Sampler]
    batch_sampler: Optional[Sampler]


def build_bhsd_transforms(patch_size, extended_aug=True): ...


def build_bhsd_dataloaders(
    *, cfg: Config, batch_layout: BatchLayout, dataset_name: str,
    fold: int, is_ddp: bool, rank: int, world_size: int,
    patch_size: Tuple[int, int, int], num_classes: int,
    is_binary: bool, class_names: Sequence[str],
) -> DataLoaderBundle: ...
```

Các chữ ký trong tài liệu là khung thiết kế; khi triển khai phải khai báo đầy đủ imports/type annotations phù hợp phiên bản Python dự án.

`setup_dataloaders()` gọi factory và gán kết quả vào các thuộc tính hiện hữu. Hai trường sampler không dùng được trả về `None`. Test xác nhận loop vẫn chọn đúng sampler để `set_epoch()` ở cả SSL và supervised.

### 4.2. W&B table builder

```python
def build_evaluation_table(self, details, class_names):
    """Trả về Table hoặc None; không gửi log và không nhận step."""
```

Đưa toàn bộ mapping 16 cột hiện tại vào method này. Khi W&B disabled hoặc không có rows, trả `None` và không yêu cầu import SDK. Giữ hành vi cảnh báo/bỏ bảng nếu dựng bảng lỗi như hiện tại.

Coordinator đưa bảng vào `val_log['val/evaluation_table']` rồi gọi `log_metrics(val_log, step=global_step)` đúng một lần cho sự kiện validation. Giữ chính sách xử lý lỗi log của `WandbManager`, gồm `wandb_required`.

### 4.3. Checkpoint metadata builder

```python
def build_checkpoint_metadata(
    *, config, wandb_identity, validation_options,
    successful_updates, validation_details, extra_state=None,
) -> dict: ...
```

Hàm không gọi SDK, đọc file hoặc serialize model. Facade lấy config đã redact và identity theo điều kiện hiện có rồi truyền vào. Builder giữ các key `config`, `wandb_run`, `metric_policy`, `validation_inference`, `successful_updates`, `validation_metrics`, `model_spec`; giữ điều kiện xuất hiện và merge order hiện tại, không mutate input.

`BaseMethod.save_checkpoint()` tiếp tục kiểm tra main process, gọi builder, gọi serializer hiện có và upload artifact. `load_checkpoint()` vẫn ở facade. Kiểm thử roundtrip xác nhận không mất EMA, optimizer2, scaler, RNG hoặc metadata; không đưa SDK object/callback vào checkpoint.

### 4.4. Validation coordinator

Giữ `ValidationRunner` chuyên trách inference. Tạo coordinator mới mỗi lần `_handle_periodic_validation()` được gọi; không thêm `self.val_runner` tồn tại lâu dài. Runner inference vẫn được tạo mới trong `evaluate_validation_model()` để đọc dataset/options hiện tại.

Dùng các dataclass có tên trường, giới hạn đúng dữ liệu cần thiết:

- `ValidationRequest`: `snap_dir`, `phase`, `iter_num`, `global_step`, `best_dice`, `best_model_name`, `legacy_best_model_name`, `art_prefix`, `latest_pth`, `should_vis`.
- `ValidationRuntime`: `device`, `is_ddp`, `rank`, `class_names` và `world_size`.
- `ValidationCallbacks`: `validate`, `save_checkpoint`, `cloud_latest_due`, `build_visual_payload`, `get_validation_details`, `resume_training_mode`, `publish_validation_state`.
- `ValidationOutcome`: `best_dice`, `last_validation_dice`, `validation_details`.

```python
class ValidationCoordinator:
    def handle(
        self, *, request: ValidationRequest, runtime: ValidationRuntime,
        callbacks: ValidationCallbacks, writer, wandb_manager,
    ) -> ValidationOutcome: ...
```

Wiring bắt buộc trong facade: callback `publish_validation_state(dice, details)` gán `last_validation_dice` và `validation_details` về trainer trước mọi callback lưu checkpoint.

```python
callbacks = ValidationCallbacks(
    validate=self.validate,
    save_checkpoint=self.save_checkpoint,
    cloud_latest_due=self.cloud_latest_due,
    build_visual_payload=self.build_visual_payload,
    get_validation_details=lambda: getattr(self, 'validation_details', {}),
    resume_training_mode=self.model.train,
    publish_validation_state=publish_validation_state,
)
```

Chi tiết thực thi:

1. Gọi callback `validate(return_visuals=request.should_vis)` trên mọi rank. Giữ tương thích kết quả hai hoặc ba phần tử ở nhánh không yêu cầu visuals.
2. Đọc details sau callback. Giữ cập nhật details trên trainer trước các callback save; không chỉ giữ bản sao cục bộ khiến checkpoint thiếu iteration/phase. Không phụ thuộc subclass phải tự tạo details khi nó override validation đơn giản.
3. Kiểm tra metrics, ghi `last_validation_dice`, chọn best và thực hiện side effects theo mục 3.3. Callback save phải nhận đầy đủ `checkpoint_kind`, `selection_metric`, `selected_iter` như hiện tại.
4. Giữ các collective theo cùng thứ tự trên mọi rank; không đặt toàn bộ coordinator trong nhánh rank 0. Visual request có thể khác giữa ranks như hiện tại nhưng không được làm lệch collective inference.
5. Gọi callback đưa model về train ở vị trí hiện tại, sau sync; trả outcome. Facade gán trạng thái outcome và trả float best như API cũ.

Không hứa xử lý được mọi lỗi một-rank: lỗi I/O rank 0 trước collective có thể khiến rank khác chờ đến timeout trong thiết kế hiện tại. Thay đổi failure protocol DDP phải là công việc riêng, với test lỗi đa tiến trình và timeout hữu hạn; không lén thay đổi trong refactor này.

### 4.5. System info và progress

`utils/system_info.py` nhận repo root tường minh; không suy ra root theo vị trí module mới. Giữ Git revision/dirty fallback, Python/PyTorch versions, batch layout, validation policy, split fingerprints, intensity metadata và actual labeled count. Facade vẫn gán `cfg.resolved` trước khi ghi config đã redact.

Tách formatter vào `utils/logger.py`; formatter nhận elapsed time, VRAM và dữ liệu progress, trả chuỗi. Wrapper `_log_iteration_progress()` vẫn đo thời gian, đọc VRAM, log và trả timer mới kiểu float. Giữ công thức ETA hiện tại trong commit extraction; sửa công thức nếu cần ở thay đổi riêng.

## 5. Kế hoạch triển khai và cổng kiểm chứng

Mỗi bước là thay đổi độc lập có thể review/revert. Chỉ chuyển bước khi checks liên quan đạt hoặc có blocker môi trường được ghi rõ. Không đồng thời sửa thuật toán trong commit extraction.

### Bước 0 — Baseline và lỗi cleanup

1. Ghi revision, working-tree status, Python/PyTorch, lệnh test, số pass/fail/skip và thời gian chạy.
2. Chạy suite hiện tại để phân biệt lỗi có sẵn với hồi quy. Không dùng số test cố định: kiểm đếm tĩnh hiện thấy 103 hàm `test_*`, không tương đương số test thực thi thành công.
3. Thêm contract tests còn thiếu trước khi chuyển logic: dispatch validation/checkpoint qua override, payload gộp, trạng thái validation trong checkpoint và phase lifecycle.
4. Sửa riêng lỗi thiếu `import sys` ở nhánh exception của `BaseMethod.train()`; test xác nhận exception gốc được ném lại, writer đóng và W&B nhận `exit_code=1` khi cleanup thành công.

### Bước 1 — Data factory

1. Tạo factory và bundle theo mục 4.1; giữ registry và các mặc định hiện tại.
2. Thay thân `setup_dataloaders()` bằng delegation.
3. `test_batching.py` hiện patch `methods.base_method.DATASETS.build`. Khi import chuyển ownership, đổi mock target sang module thực sự sử dụng registry; giữ assertion cũ.
4. Bổ sung test trực tiếp factory cho SSL/supervised, sampler và data partition nếu test hiện có chưa bao phủ.

Cổng kiểm chứng: `tests.test_batching`, `tests.test_logic_audit`, `tests.test_framework` và test factory mới.

### Bước 2 — Helper W&B, system info và progress

1. Tách table builder, không tách validation thành nhiều lần log.
2. Tách system info, truyền repo root và giữ `cfg.resolved`/redaction.
3. Tách formatter, giữ wrapper và return type.

Cổng kiểm chứng: `tests.test_validation_tracking`, `tests.test_logic_audit`, `tests.test_framework`; test builder không gửi log, disabled không import SDK, đủ 16 cột; test resolved config giữ schema và không lộ trường đã redact.

### Bước 3 — Checkpoint metadata

1. Tạo metadata builder; giữ local save/artifact orchestration trong facade.
2. Giữ serializer và manifest hiện có; không sửa schema hoặc load semantics.
3. Kiểm tra extra-state precedence và checkpoint được lưu qua override của Mean Teacher/CPS.

Cổng kiểm chứng: `tests.test_checkpoint_resolution`, `tests.test_logic_audit`, `tests.test_phase2_amp_ddp`, cùng test metadata mới.

### Bước 4 — Validation coordinator

1. Tạo request/runtime/callback/outcome và coordinator.
2. Di chuyển orchestration từng khối, giữ thứ tự side effects và hook dispatch.
3. Giữ chữ ký `_handle_periodic_validation()` và các visual wrappers.
4. Giữ runner inference hiện có, không cache runner giữa phase.

Cổng kiểm chứng: `tests.test_logic_audit`, `tests.test_validation_tracking`, `tests.test_distributed_logic`, `tests.test_phase4_metrics`, cùng contract tests coordinator mới. DDP test phải dùng collective thật với timeout hữu hạn.

### Bước 5 — Review loop và xác nhận toàn diện

1. Chỉ bỏ imports/local variables thực sự không còn sử dụng sau extraction.
2. Review `run_training_loop()`; mặc định giữ resume/phase lifecycle tại đây. Nếu vẫn cần tách resume helper, lập giao diện, contract tests và commit riêng trước khi chuyển.
3. Chạy full suite; thực hiện smoke test training ngắn qua fixtures CPU hiện có và GPU/AMP nếu môi trường hỗ trợ.
4. So sánh payload/schema/callback order và kết quả deterministic fixtures trước/sau. Golden loss tests không thay thế kiểm chứng orchestration.

## 6. Ma trận nghiệm thu

| Nhóm | Kịch bản phải kiểm chứng | Coverage sử dụng/bổ sung |
| --- | --- | --- |
| Data | SSL/supervised, local/global counts, partitions, overlap, thiếu pool, labeled-first, sampler epoch | `test_batching`, `test_logic_audit`, factory tests |
| Hook dispatch | Validation thật sự dùng EMA; save đi qua override và giữ `ema_net`/`net2`/`opt2` | Contract tests mới và `test_logic_audit` |
| Metrics | Dice 0 là best, tie không ghi đè, nonfinite bị từ chối, class mapping | `test_logic_audit`, `test_phase4_metrics` |
| Logging | Một validation payload có metrics/visual/table, đúng step, disabled và lỗi builder | `test_validation_tracking` và coordinator tests |
| Validation state | Details có iteration/global step/phase trước save; outcome cập nhật trainer | Coordinator tests mới |
| Checkpoint | Roundtrip model/optimizers/scaler/RNG/counters, extra-state precedence, manifest và redaction | `test_phase2_amp_ddp`, `test_checkpoint_resolution`, metadata tests |
| Resume/phase | Local resume, cloud fallback, missing latest, existing files, completed phase, PICK offset/allow-new-phase | `test_logic_audit` và phase tests mới |
| Final validation | Có validation cuối khi cần, latest đúng final iteration, best không bị ghi đè bởi weights chưa đánh giá | `test_logic_audit` |
| DDP | Hai rank, rank không có case, best đồng nhất, rank 0 side effects, model về train | `test_distributed_logic` và assertion bổ sung |
| Optimization | AMP overflow không partial update, EMA sau successful step, LeFeD số backward/update không đổi | `test_phase2_amp_ddp`, `test_logic_audit` |
| Lifecycle | Writer/W&B đóng đúng lúc, PICK không kết thúc session giữa phase, lỗi gốc không bị che | Lifecycle tests mới |
| API | Chữ ký/default/return của facade còn nguyên, visual wrappers còn dùng được | Contract tests và review diff |

Lệnh kiểm chứng tiêu chuẩn, chạy từ repository root:

```bash
python -m unittest tests.test_batching tests.test_logic_audit tests.test_framework
python -m unittest tests.test_validation_tracking tests.test_phase4_metrics
python -m unittest tests.test_checkpoint_resolution tests.test_phase2_amp_ddp
python -m unittest tests.test_distributed_logic
python -m unittest discover -s tests -v
```

Thêm module test mới vào cổng kiểm chứng tương ứng. Phân biệt test bị skip vì thiếu SDK/opt-in, test lỗi dependency và test pass. CPU/Gloo không chứng minh CUDA/NCCL hay AMP thực tế đã đạt; ghi rõ phần chưa chạy. Không kết luận “toàn bộ đạt” chỉ từ đếm test definitions.

## 7. Tiêu chí hoàn thành và bàn giao

- Các phần được tách có ownership rõ, không circular import và không nhận trainer nguyên khối.
- Facade/API, hook dispatch, state mutation và lifecycle được giữ theo mục 3.
- Có bằng chứng kiểm chứng cho ma trận mục 6; lỗi tồn tại trước refactor được ghi riêng.
- Không thay đổi config schema, checkpoint schema, metric names, artifact naming hoặc augmentation/optimization policy ngoài lỗi cleanup đã tách riêng.
- Không còn `wandb.Table` trực tiếp trong `BaseMethod`; data construction và metadata assembly đã nằm ở helper tương ứng.
- Báo cáo số dòng trước/sau để tham khảo, không dùng làm điều kiện đạt.
- Bàn giao danh sách file thay đổi, lệnh/kết quả test, pass/fail/skip, hạn chế môi trường và các công việc được hoãn.

Nếu một bước làm lệch contract, sửa hoặc revert riêng bước đó trước khi tiếp tục. Việc xóa wrapper deprecated, đổi DDP failure protocol, cải thiện ETA, cache signature hoặc đảm bảo exact-resume trajectory cần đề xuất và kiểm chứng riêng.
