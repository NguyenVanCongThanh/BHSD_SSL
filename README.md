# MedSSL-BHSD

A modular, extensible Semi-Supervised Learning (SSL) framework for 3D medical image segmentation, developed and benchmarked on the Brain Hemorrhage Segmentation Dataset (BHSD).

MedSSL-BHSD provides standardized training pipelines, sliding-window evaluation, capability-checked architectural contracts, and automated distributed/Kaggle execution across multiple SSL paradigms.

---

## Key Highlights

- **Standardized SSL Algorithms:** Native implementations of **Mean Teacher**, **PICK** (Pixel-Level & Consistency-Guided Knowledge), **CPS** (Cross Pseudo Supervision), **BCP** (Bidirectional CutMix Pseudo-labeling), **LeFeD** (Feedback-Driven SSL), and a **Supervised Baseline**.
- **Diverse 3D Backbones:** Support for **V-Net**, **PICK-VNet**, **3D U-Net**, **UNETR** (Transformer-based), **nnU-Net 3D** (with deep supervision), and **LeFeDNet** (dual-feedback architecture).
- **Architectural Capability Contracts:** Backbones declare functional capabilities (`segmentation`, `deep_supervision`, `mim`, `aux`, `dual_feedback`), and SSL methods declare prerequisite requirements. Incompatible pairings are intercepted before training begins.
- **Production-Grade Evaluation:** 3D sliding-window inference with Gaussian patch blending, optional 3D Test-Time Augmentation (TTA), multi-class Dice and physical surface metrics (HD95, ASD), and evaluation in both preprocessed HDF5 and native patient NIfTI space.
- **Distributed Training & Cloud Automation:** Single-GPU, PyTorch DataParallel, and multi-GPU Distributed Data Parallel (DDP via `torchrun`). Turnkey tooling for automated multi-account Kaggle submissions and dual-GPU execution.
- **Resilient Checkpointing & W&B Tracking:** Artifact-level state tracking, offline run reconciliation, and seamless crash recovery with strict split fingerprint verification.

---

## Model & Method Capability Matrix

Methods declare required capabilities; backbones declare provided capabilities. Pairing is validated dynamically at initialization:

| Method | Supported Backbones | Required Capabilities | Key Features |
|---|---|---|---|
| `supervised` | VNet, UNet3D, UNETR, nnUNet3D | `segmentation` | Supervised baseline with Dice + Cross-Entropy |
| `mean_teacher` | VNet, UNet3D, UNETR, nnUNet3D | `segmentation` | Exponential Moving Average (EMA) teacher with consistency loss; optional deep supervision |
| `cps` | VNet, UNet3D, UNETR, nnUNet3D | `segmentation` | Cross Pseudo Supervision between two independently initialized sub-networks |
| `bcp` | VNet, UNet3D, UNETR, nnUNet3D | `segmentation` | Bidirectional CutMix with dynamic pseudo-label blending |
| `pick` | PICK-VNet | `segmentation`, `mim`, `aux` | Masked Image Modeling (MIM) pre-training + feature consistency self-training |
| `lefed` | LeFeDNet | `segmentation`, `dual_feedback` | Dual-feedback network with rectified pseudo-labeling |

---

## Installation

### Prerequisites
- Linux OS (recommended)
- Python 3.10+
- CUDA-compatible GPU (recommended for training and native-space evaluation)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/NguyenVanCongThanh/BHSD_SSL.git
cd BHSD_SSL

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core training & evaluation dependencies
pip install -r requirements.txt

# (Optional) Install analysis, W&B dashboard, and profiling dependencies
pip install -r requirements-tools.txt
```

> [!NOTE]
> `requirements.txt` specifies tested lower-bound dependency versions. Always ensure your `torch` and `torchvision` builds match your local CUDA runtime.

---

## Dataset Structure

MedSSL-BHSD is designed around 3D CT volumes stored as HDF5 files (`.h5`), accompanied by split lists:

```text
/path/to/BHSD_fold0/
├── data/
│   ├── case_001.h5          # Contains datasets: "image" [Z, Y, X], "label" [Z, Y, X]
│   ├── case_002.h5
│   └── ...
├── train_labeled.list       # Case IDs for labeled training partition
├── train_unlabeled.list     # Case IDs for unlabeled training partition
├── val.list                 # Validation cases
└── test.list                # Test cases
```

### Supported Tasks
- **Binary Segmentation (`--task binary`):** Background (0) vs Intracranial Hemorrhage (1).
- **Multiclass Segmentation (`--task multiclass`):** 6 classes:
  - `0`: Background
  - `1`: Epidural Hemorrhage (EDH)
  - `2`: Intraparenchymal Hemorrhage (IPH)
  - `3`: Intraventricular Hemorrhage (IVH)
  - `4`: Subarachnoid Hemorrhage (SAH)
  - `5`: Subdural Hemorrhage (SDH)

---

## Quickstart

### 1. Training

#### Option A: Using Preset Experiment Configs
```bash
python train.py --config configs/experiments/bhsd_mt_swdl_setting_fold0.yaml \
  --data.root_path /path/to/BHSD_fold0 \
  --experiment.exp_dir /path/to/experiments
```

Available presets in `configs/experiments/`:
- `bhsd_supervised_fold0.yaml`: Supervised baseline on V-Net
- `bhsd_mt_default_fold0.yaml`: Standard Mean Teacher
- `bhsd_mt_swdl_setting_fold0.yaml`: Mean Teacher with sliding-window validation
- `bhsd_mt_nnunet_fold0.yaml`: Mean Teacher on nnU-Net 3D with deep supervision
- `bhsd_pick_fold0.yaml`: PICK two-stage framework
- `bhsd_pick_swdl_setting_fold0.yaml`: PICK with sliding-window evaluation
- `bhsd_cps_fold0.yaml`: Cross Pseudo Supervision on V-Net

#### Option B: Dynamic CLI Composition
Mix and match datasets, methods, models, and folds directly from the command line:
```bash
python train.py --dataset bhsd_multiclass --method cps --model vnet --fold 0 \
  --data.root_path /path/to/BHSD_fold0 \
  --experiment.exp_dir /path/to/experiments \
  --experiment.name cps_vnet_fold0
```

#### Option C: Multi-GPU Distributed Data Parallel (DDP)
```bash
torchrun --nproc_per_node=2 train.py --config configs/experiments/bhsd_mt_swdl_setting_fold0.yaml \
  --ddp --data.root_path /path/to/BHSD_fold0
```

#### Option D: Backward-Compatible PICK Scripts
```bash
python BHSD_multiclass_PICK_train.py --fold 0 --gpu 0 --data.root_path /path/to/BHSD_fold0
```

---

### 2. Evaluation & Benchmarking

#### Single Checkpoint Evaluation
Evaluate a trained model using 3D sliding-window inference:
```bash
python evaluate.py --dataset bhsd_multiclass \
  --checkpoint /path/to/experiments/best_model.pth \
  --root_path /path/to/BHSD_fold0 \
  --patch_size 32 192 192 \
  --metrics_output evaluation_results.json
```

Key evaluation options:
- `--use_tta`: Enable 3D Test-Time Augmentation (flips across all spatial axes).
- `--no_gaussian`: Disable Gaussian weighting during patch stitching (uses uniform averaging).
- `--surface_metrics`: Compute physical boundary metrics (HD95 and ASD).
- `--eval_space native --raw_gt_dir /path/to/raw_nifti`: Resample predictions back to native patient NIfTI space for strict nnU-Net benchmark compliance.

#### Computational Complexity Profiling
Measure model parameters, FLOPs, latency, and peak GPU memory:
```bash
python evaluate.py --model unetr --profile --patch_size 16 64 64
```

#### 5-Fold Benchmark Evaluation
Run complete cross-validation evaluation across all 5 folds:
```bash
python -m evaluation.benchmark --exp_dir /path/to/experiments/cps_vnet \
  --dataset bhsd_multiclass \
  --root_path /path/to/BHSD_data \
  --patch_size 32 192 192
```

---

## Checkpoint & Resume System

The training framework supports automated checkpoint retention and recovery:
- **Local Checkpoints:** Saves `best_model.pth`, `checkpoint_latest.pth`, and `checkpoint_epoch_latest.pth`.
- **W&B Cloud Sync:** When `logging.wandb_upload_checkpoints: true`, checkpoints and metric summaries are synced as versioned artifacts.
- **Crash Recovery:** Use `--resume` to continue training. The framework verifies split fingerprints and optimizer states before resuming iteration counts.

```bash
python train.py --config configs/experiments/bhsd_mt_swdl_setting_fold0.yaml \
  --resume --experiment.resume_policy auto
```

---

## Kaggle Deployment & Automation

The repository includes a suite of tools for running headless dual-GPU experiments on Kaggle:

1. **Manage Credentials Securely:**
   ```bash
   python -c "from tools.kaggle.accounts_manager import AccountsManager; m = AccountsManager(); m.add_account('acc1', 'username', 'key')"
   ```
2. **Generate Kaggle Notebooks:**
   ```bash
   python tools/generate_kaggle_notebook.py
   ```
3. **Submit 5-Fold Experiments:**
   ```bash
   bash scripts/submit_multiclass_5folds.sh
   ```

---

## Testing & Quality Assurance

The codebase includes an extensive suite of unit tests, regression tests, and numerical verification checks:

```bash
# Run all unit and integration tests
python -m unittest discover -s tests -v

# Run numerical golden baseline checks (Dice, CE, Consistency losses)
python -m unittest tests.test_golden_baseline -v

# Run batching & contract verification
python -m unittest tests.test_batching -v
```

---

## Project Structure

```text
BHSD_SSL/
├── configs/                 # Hierarchical YAML configs
│   ├── datasets/            # Dataset definitions (bhsd_binary, bhsd_multiclass)
│   ├── methods/             # Method configs (supervised, mean_teacher, pick, cps, bcp, lefed)
│   ├── models/              # Model configurations (vnet, unet3d, unetr, nnunet3d)
│   └── experiments/         # Complete 5-fold benchmark presets
├── core/                    # Core infrastructure
│   ├── config_parser.py     # Hierarchical config loading & CLI overrides
│   ├── model_builder.py     # Capability validation & model factories
│   ├── batching.py          # Labeled/unlabeled batch construction & pairing
│   └── distributed.py       # DDP and multi-GPU utilities
├── data/                    # Data loaders, samplers, and 3D transforms
├── evaluation/              # Sliding-window inference, metrics & 5-fold benchmark
├── losses/                  # Loss functions (Dice, Cross-Entropy, Consistency, Deep Supervision)
├── methods/                 # SSL method training loops and validation coordinators
├── models/                  # 3D backbones and architecture definitions
├── tools/                   # Kaggle submission tools, profilers, visualizer generator
│   └── archive/             # Archived one-off analysis scripts
├── scripts/                 # Shell automation and submission scripts
├── tests/                   # 160+ automated unit and integration tests
├── train.py                 # Unified training entrypoint
└── evaluate.py              # Unified evaluation and profiling entrypoint
```

---

## Detailed Documentation

For in-depth guides and architectural design specifications, refer to [docs/README.md](docs/README.md):
- [Training & Optimization Guide](docs/guides/training.md)
- [Checkpoint & Resume Policy](docs/guides/checkpoint-resume.md)
- [Checkpoint Evaluation Guide](docs/guides/checkpoint-evaluation.md)
- [Benchmark Protocol & Comparison Standards](docs/guides/benchmark-protocol.md)
- [Metrics & W&B Tracking](docs/guides/metrics-and-wandb.md)
- [Validation Visualization Guide](docs/guides/validation-visualization.md)
- [Kaggle Operations & Troubleshooting](docs/guides/kaggle.md)

---

## License

This project is licensed under the Apache 2.0 License.
