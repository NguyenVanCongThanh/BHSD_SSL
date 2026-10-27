# Audit: Benchmark 5-Fold không tìm đúng checkpoint

**Ngày audit:** 2026-10-27
**Commit tham chiếu:** `1bdeb62`
**Phạm vi:** `evaluation/benchmark.py`, `evaluate.py`, `methods/base_method.py`, `methods/pick.py`
**Mức độ:** Cao — benchmark vẫn chạy nhưng có thể đánh giá sai checkpoint hoặc bỏ qua toàn bộ fold mà không báo lỗi rõ ràng.
**Nguồn:** ghi lại chi tiết vấn đề #3 trong bản tổng quan source code ngày 27/10/2026.


> **Cập nhật triển khai 27/10/2026:** Phần audit bên dưới ghi lại trạng thái tại
> commit `1bdeb62`; các ví dụ API/manifest ở đó là đề xuất lịch sử.
> Quy ước đã triển khai và kết quả kiểm tra nằm trong
> [BENCHMARK_CHECKPOINT_POLICY.md](BENCHMARK_CHECKPOINT_POLICY.md).
> Resolver, strict loading, config-based model building, manifest khi lưu và test
> đã được bổ sung. Chưa chạy lại suy luận benchmark trên dataset.
> **Đính chính:** config trong checkpoint Mean Teacher fold 0–1 là `unet3d`;
> task là binary, 2 lớp và mới chạy 1–2 iteration; các lệnh multiclass
> `--model nnunet3d` bên dưới không phù hợp với hai checkpoint này.

---

## 1. Tóm tắt (TL;DR)

`benchmark_5folds()` tìm checkpoint theo tên:

```text
{model_name}_best_model.pth
```

Trong khi training engine lưu:

```text
best_model.pth
VNet_best_model.pth
checkpoint_latest.pth
```

Do đó:

1. **Không bao giờ tìm thấy file best thật** vì tên không khớp.
2. Nếu có `checkpoint_latest.pth`, benchmark **âm thầm đánh giá latest** thay vì best.
3. Nếu không có `checkpoint_latest.pth`, benchmark **bỏ qua fold** và có thể kết thúc với `[ERROR] Không tìm thấy kết quả của fold nào!`.
4. Trên dữ liệu hiện có của PICK, cả 5 fold đều bị bỏ qua dù đã train xong và có `VNet_best_model.pth`.
5. Với Mean Teacher hiện có, benchmark sẽ đánh giá `checkpoint_latest.pth`, không phải best.

Đây là lỗi **silent failure**: không exception, không exit code lỗi, nên rất dễ bị bỏ qua.

### Trạng thái xử lý

| Hạng mục | Trạng thái |
|---|---|
| Sửa `benchmark.py` | ☐ Chưa |
| Thêm `resolve_checkpoint()` | ☐ Chưa |
| Thêm manifest | ☐ Chưa |
| Thêm unit test | ☐ Chưa |
| Chạy lại benchmark PICK 5 fold | ☐ Chưa |
| Chạy lại benchmark Mean Teacher | ☐ Chưa |
| Cập nhật tài liệu benchmark | ☐ Chưa |

> Khi hoàn thành mục nào, đánh dấu `☑` và ghi commit/PR tương ứng vào cột bên cạnh (có thể bổ sung cột “Commit/PR” khi bắt đầu sửa).

---

## 2. Cách training engine lưu checkpoint

### 2.1. Logic lưu best

Trong `methods/base_method.py`, hàm `_handle_periodic_validation()`:

```python
if val_mdice > best_dice:
    best_dice = val_mdice
    best_pth = os.path.join(snap_dir, best_model_name)
    self.save_checkpoint(path=best_pth, ...)

    if legacy_best_model_name and legacy_best_model_name != best_model_name:
        legacy_pth = os.path.join(snap_dir, legacy_best_model_name)
        self.save_checkpoint(path=legacy_pth, ...)
```

Với mặc định:

```python
best_model_name = "best_model.pth"
legacy_best_model_name = "VNet_best_model.pth"
```

### 2.2. Logic lưu latest

Sau mỗi validation, và mỗi `save_interval` iteration:

```text
checkpoint_latest.pth
```

### 2.3. Logic lưu cuối training

Cuối `run_training_loop()`:

```python
self.save_checkpoint(latest_pth, ...)

best_pth = os.path.join(snap_dir, best_model_name)
if not os.path.exists(best_pth):
    self.save_checkpoint(best_pth, ...)
```

Lưu ý: nếu chưa từng có validation nào cải thiện, file `best_model.pth` vẫn được tạo ở cuối nhưng thực chất là trọng số cuối, không phải best theo validation.

### 2.4. Thư mục thực tế

**Mean Teacher / Supervised / BCP / CPS / LeFeD:**

```text
<exp_dir>/
└── fold_0/
    ├── best_model.pth
    ├── VNet_best_model.pth
    ├── checkpoint_latest.pth
    └── log/
```

**PICK (2 phase):**

```text
<exp_dir>/
└── fold_0/
    ├── pre_train/
    │   ├── best_model.pth
    │   ├── VNet_best_model.pth
    │   ├── checkpoint_latest.pth
    │   └── log/
    └── self_train/
        ├── best_model.pth
        ├── VNet_best_model.pth
        ├── checkpoint_latest.pth
        └── log/
```

Xem thêm:

- `methods/base_method.py` — `run_training_loop()`, `_handle_periodic_validation()`
- `methods/pick.py` — `run_pre_train()`, `run_self_train()`

---

## 3. Cách benchmark tìm checkpoint

Trong `evaluation/benchmark.py`, hàm `benchmark_5folds()`:

```python
candidate_paths = [
    os.path.join(exp_dir, f"fold_{fold}", "self_train", f"{model_name}_best_model.pth"),
    os.path.join(exp_dir, f"fold_{fold}", f"{model_name}_best_model.pth"),
    os.path.join(exp_dir, f"fold_{fold}", "self_train", "checkpoint_latest.pth"),
    os.path.join(exp_dir, f"fold_{fold}", "checkpoint_latest.pth"),
    os.path.join(exp_dir, f"fold{fold}", "self_train", f"{model_name}_best_model.pth"),
]

ckpt_path = None
for cand in candidate_paths:
    if os.path.exists(cand):
        ckpt_path = cand
        break

if ckpt_path is None:
    logging.warning(f"[Fold {fold}] Checkpoint not found in candidates, skipping fold {fold}!")
    continue
```

### Vấn đề

- Không có `best_model.pth` trong danh sách.
- Không có `VNet_best_model.pth` trong danh sách.
- Tên `{model_name}_best_model.pth` không được engine nào tạo ra.
- Đường dẫn `fold{fold}` (không có dấu gạch dưới) là dạng cũ, không khớp `fold_0`.
- Không xét `pre_train/` (có thể chấp nhận được cho benchmark cuối, nhưng cần ghi rõ).
- Đã in đường dẫn được chọn; chưa ghi loại best/latest và lý do fallback.
- Không kiểm tra metadata `best_dice` để xác nhận đó là best.

---

## 4. Phân tích 3 điểm lệch

### 4.1. Lệch tên file

| Benchmark tìm | Engine lưu | Khớp? |
|---|---|---|
| `vnet_best_model.pth` | `best_model.pth` | ❌ |
| `vnet_best_model.pth` | `VNet_best_model.pth` | ❌ |
| `pick_best_model.pth` | `best_model.pth` | ❌ |
| `nnunet3d_best_model.pth` | `best_model.pth` | ❌ |

### 4.2. Lệch hoa/thường

Trên Linux filesystem phân biệt hoa thường:

```text
vnet_best_model.pth  ≠  VNet_best_model.pth
```

Ngay cả khi chỉ sửa tên thành `model_name + "_best_model.pth"`, trường hợp `vnet` vẫn không khớp `VNet_best_model.pth`.

### 4.3. Lệch thư mục / phase

- PICK lưu best trong `self_train/` và `pre_train/`.
- Benchmark chỉ xét `self_train/` và root của fold.
- Với benchmark cuối, nên ưu tiên `self_train/`, nhưng cần xử lý rõ nếu chỉ có `pre_train/`.

---

## 5. Bằng chứng trên máy hiện tại

Mô phỏng đúng logic `candidate_paths` trên các experiment đang có:

### 5.1. PICK multi-class 5 folds

```text
/home/u001015/experiments/BHSD_multiclass_PICK_5folds/fold_0/
├── pre_train/VNet_best_model.pth
└── self_train/VNet_best_model.pth
```

Kết quả mô phỏng:

```text
fold 0: không có candidate nào khớp
fold 1: không có candidate nào khớp
fold 2: không có candidate nào khớp
fold 3: không có candidate nào khớp
fold 4: không có candidate nào khớp
```

→ `benchmark_5folds()` sẽ `continue` cả 5 fold, sau đó:

```text
[ERROR] Không tìm thấy kết quả của fold nào!
```

và **không tạo `benchmark_summary.json`**.

### 5.2. Mean Teacher

```text
/home/u001015/experiments/BHSD_mean_teacher_5folds/fold_0/checkpoint_latest.pth   (315 MB)
/home/u001015/experiments/BHSD_mean_teacher_5folds/fold_1/checkpoint_latest.pth   (315 MB)
```

Kết quả mô phỏng:

```text
fold 0: chỉ tìm thấy checkpoint_latest.pth
fold 1: chỉ tìm thấy checkpoint_latest.pth
fold 2–4: không có
```

→ Benchmark sẽ đánh giá latest ở fold 0 và 1, bỏ qua fold 2–4.

### 5.3. Supervised

```text
/home/u001015/experiments/BHSD_supervised_baseline/fold_0/checkpoint_latest.pth   (76 MB)
```

→ Benchmark đánh giá latest, không phải best.

> Ghi chú: các file `VNet_best_model.pth` hiện có từ đợt chạy tháng 9/2026, có thể được tạo bởi code cũ. Tuy nhiên, với code hiện tại, engine vẫn tạo `best_model.pth` và `VNet_best_model.pth`, nên lỗi tên file vẫn còn nguyên.

---

## 6. Ảnh hưởng tới kết quả báo cáo

### 6.1. Đánh giá latest thay vì best

Ví dụ từ log PICK trên máy:

| Fold | Best mDice | mDice tại iteration cuối |
|---|---:|---:|
| 0 | 0.5526 | 0.3658 |
| 1 | 0.5519 | 0.4442 |
| 2 | 0.5564 | 0.4657 |
| 3 | 0.6115 | 0.5952 |
| 4 | 0.5368 | 0.4309 |

Nếu benchmark lấy `checkpoint_latest.pth`, kết quả OOF có thể thấp hơn đáng kể so với best. Điều này đặc biệt nguy hiểm khi so sánh với nnUNet hoặc công bố kết quả.

### 6.2. Bỏ fold

Nếu best không được tìm thấy và latest không tồn tại (như PICK hiện tại), fold bị loại khỏi benchmark. Kết quả OOF khi đó:

- Có thể chỉ dựa trên 1–2 fold.
- Không phản ánh đúng 5-fold cross-validation.
- Dễ bị hiểu nhầm là “chưa train xong”.

### 6.3. Không có dấu vết kiểm toán

`benchmark_summary.json` chỉ ghi `checkpoint` cho từng fold nếu fold đó chạy thành công. Nếu tất cả fold bị bỏ qua, file không được tạo. Không có log tổng hợp “đã dùng best hay latest”.

---

## 7. Nguyên nhân gốc

1. **Không có single source of truth cho việc đặt tên checkpoint.**
   Engine lưu một kiểu, benchmark tìm một kiểu, tài liệu lại ghi một kiểu.

2. **Fallback quá im lặng.**
   `checkpoint_latest.pth` được ưu tiên sau best, nhưng vì best không bao giờ khớp, latest trở thành mặc định trên thực tế.

3. **Không kiểm tra metadata.**
   Checkpoint có `best_dice` và `iter_num`, nhưng benchmark không đọc để xác nhận.

4. **Không có test cho checkpoint resolution.**
   `tests/` hiện chỉ test batch, framework, crawl storage; không test việc tìm checkpoint.

5. **Model architecture không được suy ra từ experiment.**
   `evaluate.py --exp_dir` yêu cầu `--model` thủ công, mặc định `vnet`. Nếu người dùng quên, có thể build sai kiến trúc và load checkpoint với `strict=False` (lỗi liên quan, xem mục 11).

---

## 8. Hướng khắc phục

### 8.1. Cách ngắn hạn (ít rủi ro)

Bổ sung đúng tên file thật vào candidate, ưu tiên best trước latest:

```python
def _checkpoint_candidates(exp_dir, fold, model_name):
    fold_dir = os.path.join(exp_dir, f"fold_{fold}")
    phase_dirs = [
        os.path.join(fold_dir, "self_train"),
        fold_dir,
    ]
    best_names = [
        "best_model.pth",
        f"{model_name}_best_model.pth",
        "VNet_best_model.pth",
    ]
    latest_names = ["checkpoint_latest.pth"]

    candidates = []
    for d in phase_dirs:
        for n in best_names:
            candidates.append(os.path.join(d, n))
    for d in phase_dirs:
        for n in latest_names:
            candidates.append(os.path.join(d, n))
    return candidates
```

Sau đó log rõ:

```python
print(f"[Fold {fold}] Dùng checkpoint: {ckpt_path} (kind={kind})")
```

### 8.2. Cách trung hạn — helper dùng chung

Tạo `evaluation/checkpoint_resolver.py` (hoặc `utils/checkpoint.py`) với API:

```python
def resolve_checkpoint(
    exp_dir: str,
    fold: int,
    model_name: str,
    prefer: str = "best",   # "best" | "latest"
    include_pretrain: bool = False,
) -> tuple[str, str]:
    """
    Trả về (path, kind) với kind là "best" hoặc "latest".
    Ưu tiên best; nếu không có best thì fallback latest kèm warning.
    """
```

Logic đề xuất:

1. Xác định các thư mục theo thứ tự:
   - `fold_x/self_train`
   - `fold_x`
   - `fold_x/pre_train` (chỉ khi `include_pretrain=True`)
2. Tìm các file best:
   - `best_model.pth`
   - `VNet_best_model.pth`
   - `{model_name}_best_model.pth`
3. Chọn phase trước, rồi ưu tiên tên canonical trong phase; không so `best_dice` xuyên phase hoặc dùng nó để chứng minh trọng số là best.
4. Nếu không có best:
   - Nếu `prefer="best"`: log warning rõ ràng.
   - Fallback `checkpoint_latest.pth`.
5. Nếu không có gì: raise `FileNotFoundError` với danh sách đã tìm.
6. Trả thêm `kind` để caller ghi vào báo cáo.

Gợi ý đọc metadata:

```python
import torch

def _read_best_dice(path: str) -> float:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return float(ckpt.get("best_dice", -1.0))
    except Exception:
        return -1.0
```

### 8.3. Cách bền vững — manifest

Cuối mỗi phase, ghi `checkpoint_manifest.json`:

```json
{
  "phase": "self_train",
  "fold": 0,
  "model": "pick_vnet",
  "dataset": "bhsd",
  "task": "multiclass",
  "best_checkpoint": "self_train/best_model.pth",
  "best_dice": 0.5526,
  "best_iter": 11500,
  "latest_checkpoint": "self_train/checkpoint_latest.pth",
  "final_iter": 12000,
  "final_dice": 0.3658,
  "created_at": "2026-10-26T12:00:00Z"
}
```

Benchmark chỉ cần đọc manifest, không phải đoán tên file. Manifest cũng hữu ích cho resume và so sánh run.

Nên đặt manifest tại:

```text
<exp_dir>/fold_0/checkpoint_manifest.json
```

hoặc trong từng phase:

```text
<exp_dir>/fold_0/self_train/checkpoint_manifest.json
```

### 8.4. Thay đổi API benchmark

Thêm tham số:

```python
benchmark_5folds(
    ...,
    checkpoint_preference="best",   # "best" | "latest"
    include_pretrain=False,
    on_missing="error",             # "error" | "warn" | "skip"
)
```

Hành vi đề xuất:

- `on_missing="error"`: fail ngay nếu thiếu checkpoint của bất kỳ fold nào.
- `on_missing="warn"`: bỏ fold nhưng ghi rõ trong summary.
- `on_missing="skip"`: giữ hành vi cũ, nhưng có warning và summary ghi `skipped_folds`.

Mặc định nên là `"error"` để tránh báo cáo thiếu fold mà không biết.

---

## 9. Kiểm thử cần bổ sung

Tạo `tests/test_checkpoint_resolution.py`:

```python
def test_prefers_best_over_latest(tmp_path):
    fold = tmp_path / "fold_0"
    fold.mkdir()
    (fold / "best_model.pth").write_bytes(b"best")
    (fold / "checkpoint_latest.pth").write_bytes(b"latest")

    path, kind = resolve_checkpoint(str(tmp_path), 0, "vnet")
    assert kind == "best"
    assert path.endswith("best_model.pth")


def test_accepts_legacy_vnet_name(tmp_path):
    fold = tmp_path / "fold_0"
    fold.mkdir()
    (fold / "VNet_best_model.pth").write_bytes(b"best")

    path, kind = resolve_checkpoint(str(tmp_path), 0, "vnet")
    assert kind == "best"
    assert path.endswith("VNet_best_model.pth")


def test_pick_prefers_self_train(tmp_path):
    self_train = tmp_path / "fold_0" / "self_train"
    self_train.mkdir(parents=True)
    (self_train / "best_model.pth").write_bytes(b"best")
    (tmp_path / "fold_0" / "pre_train").mkdir()
    (tmp_path / "fold_0" / "pre_train" / "best_model.pth").write_bytes(b"pre")

    path, kind = resolve_checkpoint(str(tmp_path), 0, "pick_vnet")
    assert "self_train" in path


def test_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path), 0, "vnet")
```

Nếu chưa dùng `pytest`, có thể viết bằng `unittest` và `tempfile.TemporaryDirectory()` như các test hiện có.

---

## 10. Quy trình xác minh sau khi sửa

### 10.1. Kiểm tra resolver bằng dữ liệu thật

```bash
python - <<'PY'
from evaluation.checkpoint_resolver import resolve_checkpoint

cases = [
    ("/home/u001015/experiments/BHSD_multiclass_PICK_5folds", 0, "pick_vnet"),
    ("/home/u001015/experiments/BHSD_mean_teacher_5folds", 0, "nnunet3d"),
    ("/home/u001015/experiments/BHSD_supervised_baseline", 0, "vnet"),
]
for exp, fold, model in cases:
    path, kind = resolve_checkpoint(exp, fold, model)
    print(f"{exp} | fold={fold} | model={model} -> {kind}: {path}")
PY
```

Kỳ vọng:

```text
PICK fold 0     -> best: .../self_train/VNet_best_model.pth
Mean Teacher    -> latest: .../checkpoint_latest.pth (kèm warning vì thiếu best)
Supervised      -> latest: .../checkpoint_latest.pth (kèm warning vì thiếu best)
```

### 10.2. Chạy benchmark PICK

```bash
python evaluate.py \
  --model pick \
  --dataset bhsd_multiclass \
  --patch_size 32 160 160 \
  --exp_dir /home/u001015/experiments/BHSD_multiclass_PICK_5folds
```

Kỳ vọng:

- Cả 5 fold đều được đánh giá.
- Log ghi rõ checkpoint được chọn cho từng fold.
- `benchmark_summary.json` được tạo.
- Mỗi entry có `checkpoint` và `checkpoint_kind`.

### 10.3. Chạy benchmark Mean Teacher

```bash
python evaluate.py \
  --model nnunet3d \
  --dataset bhsd_multiclass \
  --patch_size 32 192 192 \
  --exp_dir /home/u001015/experiments/BHSD_mean_teacher_5folds
```

Kỳ vọng:

- Fold 0, 1 chạy với `checkpoint_latest.pth`.
- Log warning rõ: “best không tồn tại, fallback latest”.
- Fold 2–4 báo thiếu checkpoint; tùy `on_missing` mà fail hoặc ghi `skipped_folds`.

### 10.4. Kiểm tra `benchmark_summary.json`

Nên có cấu trúc tối thiểu:

```json
{
  "dataset": "bhsd",
  "task": "multiclass",
  "model": "pick",
  "evaluated_folds": [0, 1, 2, 3, 4],
  "skipped_folds": [],
  "checkpoint_preference": "best",
  "fold_results": {
    "fold_0": {
      "mDice": 0.55,
      "per_class": [0.81, 0.27, 0.23, 0.39, 0.04],
      "num_cases": 38,
      "checkpoint": "fold_0/self_train/VNet_best_model.pth",
      "checkpoint_kind": "best"
    }
  }
}
```

---

## 11. Lệnh eval tạm thời (workaround)

Trong lúc chưa sửa benchmark, eval trực tiếp từng checkpoint:

### PICK — dùng best cũ

```bash
python evaluate.py \
  --model pick \
  --dataset bhsd_multiclass \
  --patch_size 32 160 160 \
  --checkpoint /home/u001015/experiments/BHSD_multiclass_PICK_5folds/fold_0/self_train/VNet_best_model.pth \
  --fold 0
```

Lặp lại cho `fold_1` … `fold_4`.

### Mean Teacher — hiện chỉ có latest

```bash
python evaluate.py \
  --model nnunet3d \
  --dataset bhsd_multiclass \
  --patch_size 32 192 192 \
  --checkpoint /home/u001015/experiments/BHSD_mean_teacher_5folds/fold_0/checkpoint_latest.pth \
  --fold 0
```

### Supervised

```bash
python evaluate.py \
  --model vnet \
  --dataset bhsd_multiclass \
  --patch_size 32 160 160 \
  --checkpoint /home/u001015/experiments/BHSD_supervised_baseline/fold_0/checkpoint_latest.pth \
  --fold 0
```

> Quan trọng: với PICK phải dùng `--model pick` để builder tạo `pick_vnet`. Nếu dùng `--model vnet`, checkpoint PICK sẽ load thiếu decoder (`main_decoder.*` so với `decoder.*`) và cho dự đoán rác.

---

## 12. Vấn đề liên quan (cần xử lý cùng đợt)

### 12.1. Model builder bỏ qua cấu hình kiến trúc

`evaluate.py` hiện build model bằng:

```python
def model_builder_helper(name, in_channels=1, num_classes=6):
    actual_name = "pick_vnet" if name == "pick" else name
    return MODELS.build(
        actual_name,
        in_channels=in_channels,
        num_classes=num_classes,
        has_dropout=False
    )
```

Không truyền:

- `base_filters`
- `num_stages`
- `deep_supervision`
- `n_filters`
- `feat_channels`
- `embed_dim`
- `patch_size`
- `num_heads`
- `anisotropic_kernels`
- `use_strided_conv`

Hệ quả:

- `nnunet3d` luôn build với default `num_stages=7`, `base_filters=32`.
- Nếu checkpoint được train với cấu hình khác, có thể lệch kiến trúc.
- `load_checkpoint(..., strict=False)` sẽ bỏ qua key thiếu một cách im lặng.

**Đề xuất đã điều chỉnh:** engine đã lưu `config` trong checkpoint; eval cần dùng config đó và yêu cầu `--model` rõ ràng với checkpoint cũ không có config.

### 12.2. `--model` mặc định là `vnet`

`evaluate.py` mặc định `--model vnet`. Nếu chạy benchmark cho Mean Teacher mà quên `--model nnunet3d`, model sẽ được build sai. Nên:

- Suy ra model từ checkpoint metadata / manifest.
- Hoặc bắt buộc `--model` khi dùng `--exp_dir`.
- Hoặc fail nếu checkpoint và kiến trúc không khớp.

### 12.3. `checkpoint_latest.pth` không mang nghĩa “best”

Tên `latest` rõ ràng là trạng thái cuối. Việc eval latest không sai về mặt kỹ thuật, nhưng phải được ghi rõ trong báo cáo. Không nên coi latest là best mặc định.

### 12.4. `best_model.pth` có thể không phải best thật

Nếu chưa từng có validation nào cải thiện, cuối training engine vẫn tạo `best_model.pth` từ trọng số cuối. Khi đó metadata `best_dice` có thể là `0.0` hoặc giá trị cũ. `best_dice` là điểm tốt nhất lịch sử, nên kể cả giá trị dương cũng không chứng minh trọng số hiện tại là best. Engine mới chỉ tạo best khi validation thực sự cải thiện.

---

## 13. Checklist khắc phục

- [ ] Thêm `best_model.pth` và `VNet_best_model.pth` vào candidate.
- [ ] Ưu tiên `self_train/` trước `pre_train/`.
- [ ] Log rõ checkpoint được chọn và `kind` (best/latest).
- [ ] Đọc metadata `best_dice` để xác nhận best.
- [ ] Thêm helper `resolve_checkpoint()` dùng chung.
- [ ] Thêm manifest cuối training.
- [ ] Thêm `checkpoint` + `checkpoint_kind` vào `benchmark_summary.json`.
- [ ] Thêm `skipped_folds`, `evaluated_folds`.
- [ ] Thêm option `on_missing` (mặc định `error`).
- [ ] Thêm test `tests/test_checkpoint_resolution.py`.
- [ ] Sửa `evaluate.py` để build model từ config/manifest.
- [ ] Chạy lại benchmark PICK 5 fold và lưu kết quả mới.
- [ ] Chạy lại benchmark Mean Teacher (fold 0, 1) và ghi rõ là latest.
- [ ] Cập nhật tài liệu benchmark với quy ước checkpoint.

---

## 14. Tiêu chí hoàn thành

1. `evaluate.py --exp_dir` tìm được `best_model.pth` / `VNet_best_model.pth`.
2. Không còn trường hợp “có checkpoint best nhưng benchmark báo không tìm thấy”.
3. Nếu fallback latest, log phải nói rõ lý do.
4. `benchmark_summary.json` ghi đường dẫn và loại checkpoint cho từng fold.
5. Có ít nhất 4 unit test cho resolver.
6. Benchmark PICK 5 fold chạy đủ 5 fold trên dữ liệu hiện có.
7. Kết quả Mean Teacher ghi rõ là đánh giá latest, không ghi nhầm là best.

---

## 15. Phụ lục

### A. Đường dẫn code liên quan

| File | Nội dung |
|---|---|
| `evaluation/benchmark.py` | `benchmark_5folds()`, candidate paths, summary JSON |
| `evaluate.py` | CLI, `model_builder_helper()`, gọi benchmark |
| `methods/base_method.py` | lưu best/latest, `run_training_loop()` |
| `methods/pick.py` | thư mục `pre_train/`, `self_train/` |
| `utils/checkpoint.py` | `save_checkpoint()`, `load_checkpoint()` |
| `tests/` | nơi cần thêm test resolver |

### B. Log PICK tham chiếu

Nguồn: `/home/u001015/experiments/logs_pick_multi_fold_*.log`

```text
fold 0: Best mDice: 0.5526 | last val: 0.3658
fold 1: Best mDice: 0.5519 | last val: 0.4442
fold 2: Best mDice: 0.5564 | last val: 0.4657
fold 3: Best mDice: 0.6115 | last val: 0.5952
fold 4: Best mDice: 0.5368 | last val: 0.4309
```

### C. Tài liệu liên quan

- `docs/BENCHMARK_ALIGNMENT_MEAN_TEACHER_VS_NNUNET.md`
- `docs/DDP_BATCH_SIZE.md`
- `docs/EXPERIMENT_DATA.md`
- `docs/RUN_TERMINATION_AUDIT.md`

---

### D. Script tái lập kiểm tra nhanh (không cần torch)

Chạy script sau để xem, với từng experiment hiện có, benchmark sẽ tìm thấy file nào:

```python
import os

def candidates(exp_dir, fold, model_name):
    return [
        os.path.join(exp_dir, f"fold_{fold}", "self_train", f"{model_name}_best_model.pth"),
        os.path.join(exp_dir, f"fold_{fold}", f"{model_name}_best_model.pth"),
        os.path.join(exp_dir, f"fold_{fold}", "self_train", "checkpoint_latest.pth"),
        os.path.join(exp_dir, f"fold_{fold}", "checkpoint_latest.pth"),
        os.path.join(exp_dir, f"fold{fold}", "self_train", f"{model_name}_best_model.pth"),
    ]

experiments = [
    ("/home/u001015/experiments/BHSD_multiclass_PICK_5folds", "pick"),
    ("/home/u001015/experiments/BHSD_mean_teacher_5folds", "nnunet3d"),
    ("/home/u001015/experiments/BHSD_supervised_baseline", "vnet"),
]

for exp_dir, model_name in experiments:
    print("=" * 80)
    print(exp_dir, "| model =", model_name)
    for fold in range(5):
        found = [p for p in candidates(exp_dir, fold, model_name) if os.path.exists(p)]
        if found:
            print(f"  fold {fold}: {found[0]}")
        else:
            print(f"  fold {fold}: KHÔNG TÌM THẤY -> benchmark sẽ bỏ qua fold")
```

Kỳ vọng trước khi sửa:

```text
PICK:            tất cả 5 fold KHÔNG TÌM THẤY
Mean Teacher:    fold 0, 1 -> checkpoint_latest.pth; fold 2–4 -> không tìm thấy
Supervised:      fold 0 -> checkpoint_latest.pth; fold 1–4 -> không tìm thấy
```

Sau khi sửa, script nên được thay bằng `resolve_checkpoint()` ở mục 10.1 để kiểm tra đúng logic mới.

---

## 16. Kết luận

Đây không phải lỗi “thiếu một dòng code”, mà là lỗi **thiếu hợp đồng giữa nơi lưu và nơi đọc checkpoint**. Cách sửa đúng là:

1. Chuẩn hóa tên và vị trí checkpoint.
2. Tập trung hóa logic resolve.
3. Ưu tiên best có xác minh metadata.
4. Log và ghi summary rõ ràng.
5. Có test tự động.

Nếu chỉ thêm một candidate path, lỗi sẽ giảm nhưng vẫn có thể tái phát khi thêm method mới hoặc đổi cấu trúc thư mục. Manifest + resolver dùng chung là giải pháp nên hướng tới.
