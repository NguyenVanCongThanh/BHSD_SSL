# BHSD fold0 EDA revision

Canonical analysis: `notebooks/EDA_BHSD_Dataset.ipynb`.
Generator: `tools/generate_bhsd_eda_notebook.py`.
Extraction and physical-unit calculations: `tools/bhsd_eda.py`.

## Scope and reproducibility

The updated scan covers all 2,170 H5 files: 154 labeled training, 38 validation,
and 1,978 unlabeled cases. Previously, the summary CSV contained only 300 unlabeled
cases. Unlabeled zero masks are placeholders and are excluded from pathology
statistics. `test.txt` is identical to `val.txt`; it is not independent test data.
Split checks establish case-ID integrity, not patient-level independence.

Install the optional dependencies from `requirements-eda.txt` in the intended
runtime, then run:

```bash
python tools/generate_bhsd_eda_notebook.py
python -m unittest tests.test_bhsd_eda -v
```

Source paths default to `~/dataset/BHSD_fold0` and
`~/dataset/BHSD/label_192/label_192`. Override using `BHSD_DATASET_DIR` and
`BHSD_RAW_DIR`. The generator resolves output paths from its own repository, and
works independently of the caller's working directory. Per-case fingerprints use
source size/mtime and helper code SHA256. `bhsd_eda_manifest.json` records source
paths and split hashes. Delete the ignored `notebooks/eda_cache/` to force rescanning.
Existing root summary/morphology CSVs are synchronized compatibility copies;
canonical CSVs are beside the notebook.

## Source issues found

All H5 files passed the implemented shape, finite-intensity, label-range and
metadata-consistency checks. One labeled training case has no foreground and needs
annotation review. Four raw image/GT pairs have affine disagreements, so they are
listed explicitly and excluded from raw-to-H5 retention comparisons:

- `ID_089596a9_ID_ea71b57820`
- `ID_727c946e_ID_e1fd84171a`
- `ID_88dbe13f_ID_01c5078fbf`
- `ID_cb4c887c_ID_ec2a4643f3`

Among the 188 metadata-consistent raw pairs, 33 annotated voxels fall outside the
recorded crop, across three case/class pairs. Three other case/class pairs have
positive raw annotations and zero H5 voxels:

| Case | Split | Class | Raw volume (mL, recorded spacing) |
|---|---|---|---:|
| `ID_56faaf6f_ID_dd8f688dff` | validation | IPH | 0.475764 |
| `ID_6caf8d60_ID_3634a47122` | labeled train | IPH | 0.052452 |
| `ID_9ba3be10_ID_1c4e9f3c40` | validation | SAH | 0.028967 |

The first case has native 48 slices at 0.5 mm spacing and only five H5 slices at
5 mm. This is a source-metadata/preprocessing review item; the EDA does not
silently change it. A total of 29 positive case/class pairs change volume by more
than 20% under the recorded-spacing convention. See
`notebooks/bhsd_preprocessing_review_cases.csv` for per-case evidence.

## Corrections to interpretation

- Removed the contradictory crop finding derived from one case. Sampling uses the
  production `RandomCrop` on all 154 training cases, three XY sizes and foreground
  probabilities 0/0.33/0.5, with seed 42 and 40 trials per case/configuration.
  Increase `BHSD_EDA_TRIALS` for better Monte Carlo precision. Report per-case,
  class-conditional and lesion-burden results; bootstrap cases for mean intervals.
- Compare all unlabeled cases with labeled **train**, report KS effect sizes and
  Holm-adjusted p-values, and avoid claims of zero domain shift or scanner equality.
- Raw/H5 visualization reconstructs the H5 into the recorded native bbox and uses
  common voxel indices, raw affine coordinates and physical aspect. It explicitly
  assumes an inverse resampling convention; exact interpolation provenance is absent.
- Morphology now has 430 case/class unions, including the one-voxel IPH mask that
  was previously omitted. It is not a count of lesions. Component counts are
  reported separately. SAVR is recomputed as cm²/mL = cm⁻¹, fixing the factor-of-ten
  unit mismatch. Degenerate PCA is NaN. Tiny-mask sphericity can exceed one because
  voxel volume and triangulated area use different discretizations; this is flagged,
  not clipped or interpreted as a real physical shape.
- Cropping is not skull stripping; clipping saturates rather than removes tissues.
  Foreground voxel fractions use the whole cropped volume as denominator.
- Patch size, loss, task definition, foreground probability and SSL schedules are
  experiment candidates, not requirements established by EDA.

Raw labels, image arrays and training configuration were not changed. Resolving
source geometry, annotation or resampling issues requires acquisition provenance
and a separate reviewed preprocessing change.
