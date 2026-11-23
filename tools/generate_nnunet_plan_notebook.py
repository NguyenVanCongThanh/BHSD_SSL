import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kaggle.notebook_templates import (
    build_notebook_envelope,
    code_cell,
    load_notebook_presets,
    markdown_cell,
)


def build_nnunet_plan_notebook():
    presets = load_notebook_presets()

    cells = [
        markdown_cell([
            "# BHSD (Wendy-Fly/BHSD) — nnU-Net v2 Fingerprint & Experiment Planning (192 ca)\n\n",
            "Notebook này chạy `nnUNetv2_extract_fingerprint` và `nnUNetv2_plan_experiment` trực tiếp trên 192 ca NIfTI gốc từ `Wendy-Fly/BHSD` (`label_192.zip`, ~1.45 GB) để thu thập `dataset_fingerprint.json` và `nnUNetPlans_*.json` (8 GB mặc định, 15 GB & 16 GB T4) mà không tốn dung lượng máy local.\n",
        ]),
        code_cell([
            "!pip install -q nnunetv2\n",
            "import torch, nnunetv2\n",
            "print('nnunetv2:', nnunetv2.__file__, '| torch:', torch.__version__)\n",
        ]),
        code_cell([
            "import json, os, shutil, urllib.request, zipfile\n",
            "from pathlib import Path\n\n",
            "WORK = Path('/kaggle/working')\n",
            "RAW_EXTRACT = WORK / 'bhsd_raw_extract'\n",
            "NNUNET_BASE = WORK / 'nnUNet_data'\n",
            "NNUNET_RAW = NNUNET_BASE / 'nnUNet_raw'\n",
            "NNUNET_PREP = NNUNET_BASE / 'nnUNet_preprocessed'\n",
            "NNUNET_RES = NNUNET_BASE / 'nnUNet_results'\n",
            "OUT_DIR = WORK / 'nnunet_bhsd_plans'\n",
            "for d in (RAW_EXTRACT, NNUNET_RAW, NNUNET_PREP, NNUNET_RES, OUT_DIR):\n",
            "    d.mkdir(parents=True, exist_ok=True)\n\n",
            "ZIP_PATH = WORK / 'label_192.zip'\n",
            "if not ZIP_PATH.exists():\n",
            "    url = 'https://huggingface.co/datasets/Wendy-Fly/BHSD/resolve/main/label_192.zip'\n",
            "    print('Downloading', url)\n",
            "    urllib.request.urlretrieve(url, ZIP_PATH)\n",
            "print('Downloaded size (MB):', round(ZIP_PATH.stat().st_size / 1e6, 2))\n\n",
            "with zipfile.ZipFile(ZIP_PATH, 'r') as zf:\n",
            "    zf.extractall(RAW_EXTRACT)\n",
            "ZIP_PATH.unlink(missing_ok=True)\n\n",
            "# Inspect extracted structure and collect all 192 (image, label) pairs\n",
            "nii_files = sorted(RAW_EXTRACT.rglob('*.nii.gz'))\n",
            "print('Total .nii.gz found:', len(nii_files))\n",
            "for p in nii_files[:10]:\n",
            "    print(' ', p.relative_to(RAW_EXTRACT))\n\n",
            "images_by_id, labels_by_id = {}, {}\n",
            "for p in nii_files:\n",
            "    parts_lower = [x.lower() for x in p.parts]\n",
            "    stem = p.name[:-7] if p.name.endswith('.nii.gz') else p.stem\n",
            "    if stem.endswith('_0000'):\n",
            "        stem = stem[:-5]\n",
            "    if any('label' in part or 'mask' in part or 'seg' in part for part in parts_lower[:-1]):\n",
            "        labels_by_id[stem] = p\n",
            "    elif any('image' in part for part in parts_lower[:-1]):\n",
            "        images_by_id[stem] = p\n",
            "    else:\n",
            "        raise RuntimeError(f'Cannot classify image vs label path: {p}')\n\n",
            "common_ids = sorted(set(images_by_id) & set(labels_by_id))\n",
            "print(f'Matched pairs: {len(common_ids)} (images={len(images_by_id)}, labels={len(labels_by_id)})')\n",
            "assert len(common_ids) == 192, f'Expected 192 pairs, got {len(common_ids)}'\n\n",
            "ds_dir = NNUNET_RAW / 'Dataset001_BHSD'\n",
            "img_tr = ds_dir / 'imagesTr'\n",
            "lbl_tr = ds_dir / 'labelsTr'\n",
            "img_tr.mkdir(parents=True, exist_ok=True)\n",
            "lbl_tr.mkdir(parents=True, exist_ok=True)\n",
            "for cid in common_ids:\n",
            "    dst_img = img_tr / f'{cid}_0000.nii.gz'\n",
            "    dst_lbl = lbl_tr / f'{cid}.nii.gz'\n",
            "    if not dst_img.exists():\n",
            "        shutil.move(str(images_by_id[cid]), str(dst_img))\n",
            "    if not dst_lbl.exists():\n",
            "        shutil.move(str(labels_by_id[cid]), str(dst_lbl))\n",
            "shutil.rmtree(RAW_EXTRACT, ignore_errors=True)\n\n",
            "dataset_json = {\n",
            "    'channel_names': {'0': 'CT'},\n",
            "    'labels': {\n",
            "        'background': 0,\n",
            "        'epidural': 1,\n",
            "        'intraparenchymal': 2,\n",
            "        'intraventricular': 3,\n",
            "        'subarachnoid': 4,\n",
            "        'subdural': 5,\n",
            "    },\n",
            "    'numTraining': len(common_ids),\n",
            "    'file_ending': '.nii.gz',\n",
            "}\n",
            "(ds_dir / 'dataset.json').write_text(json.dumps(dataset_json, indent=2), encoding='utf-8')\n",
            "print('Prepared Dataset001_BHSD with', len(common_ids), 'cases')\n",
        ]),
        code_cell([
            "import os, subprocess\n",
            "env = os.environ.copy()\n",
            "env['nnUNet_raw'] = '/kaggle/working/nnUNet_data/nnUNet_raw'\n",
            "env['nnUNet_preprocessed'] = '/kaggle/working/nnUNet_data/nnUNet_preprocessed'\n",
            "env['nnUNet_results'] = '/kaggle/working/nnUNet_data/nnUNet_results'\n\n",
            "cmds = [\n",
            "    ['nnUNetv2_extract_fingerprint', '-d', '1', '-np', '4', '--verify_dataset_integrity'],\n",
            "    ['nnUNetv2_plan_experiment', '-d', '1', '-gpu_memory_target', '8', '-overwrite_plans_name', 'nnUNetPlans_8GB'],\n",
            "    ['nnUNetv2_plan_experiment', '-d', '1', '-gpu_memory_target', '15', '-overwrite_plans_name', 'nnUNetPlans_15GB_T4'],\n",
            "    ['nnUNetv2_plan_experiment', '-d', '1', '-gpu_memory_target', '16', '-overwrite_plans_name', 'nnUNetPlans_16GB_T4'],\n",
            "    ['nnUNetv2_plan_experiment', '-d', '1', '-pl', 'nnUNetPlannerResEncM', '-overwrite_plans_name', 'nnUNetResEncMPlans'],\n",
            "]\n",
            "for c in cmds:\n",
            "    print('>>>', ' '.join(c), flush=True)\n",
            "    subprocess.run(c, env=env, check=True)\n",
        ]),
        code_cell([
            "import json, shutil\n",
            "from pathlib import Path\n\n",
            "prep_dir = Path('/kaggle/working/nnUNet_data/nnUNet_preprocessed/Dataset001_BHSD')\n",
            "out_dir = Path('/kaggle/working/nnunet_bhsd_plans')\n",
            "for f in sorted(prep_dir.glob('*.json')):\n",
            "    shutil.copy2(f, out_dir / f.name)\n",
            "    print('Saved:', out_dir / f.name)\n\n",
            "# Print concise summary + full 3d_fullres plans to stdout for easy log inspection\n",
            "fp = json.loads((out_dir / 'dataset_fingerprint.json').read_text())\n",
            "print('\\n=== FINGERPRINT SUMMARY ===')\n",
            "print('num_cases:', len(fp.get('spacings', [])))\n",
            "print('foreground_intensity_properties_per_channel:', json.dumps(fp.get('foreground_intensity_properties_per_channel'), indent=2))\n\n",
            "for pfile in sorted(out_dir.glob('nnUNet*Plans*.json')):\n",
            "    plan = json.loads(pfile.read_text())\n",
            "    print(f'\\n=== {pfile.name} (3d_fullres) ===')\n",
            "    print(json.dumps(plan['configurations']['3d_fullres'], indent=2))\n\n",
            "# Clean heavy raw data so /kaggle/working output stays tiny\n",
            "shutil.rmtree('/kaggle/working/nnUNet_data', ignore_errors=True)\n",
            "print('Done! Remaining output files:', [p.name for p in sorted(out_dir.iterdir())])\n",
        ]),
    ]

    nb = build_notebook_envelope(cells, presets=presets)
    out_path = REPO_ROOT / "scripts" / "kaggle_bhsd_nnunet_plan.ipynb"
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated {out_path}")


if __name__ == "__main__":
    build_nnunet_plan_notebook()
