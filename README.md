# High-Desensitization GNN Workflow Repository

This repository contains a four-stage workflow for porous molecular structure analysis:

1. CSD extraction and conversion to XYZ (`scripts/csd_extractor.py`)
2. GNN model training in notebook form (`scripts/train_dimenet_schnet.ipynb`)
3. Global inference from trained weights (`scripts/dimenet_inference.py`)
4. Full t-SNE projection using inference output (`scripts/tsne_pipeline.py`)

CIF files are obtained via CCDC Python API from the CSD database (November 2025 release).

## Scope

This repository covers the GNN workflow only. It does not include external baseline-screening pipelines.

## Project Layout

- `scripts/csd_extractor.py`: CSD extraction script for exporting CIF/XYZ files.
- `scripts/train_dimenet_schnet.ipynb`: Main training notebook for DimeNet++ and SchNet.
- `scripts/dimenet_inference.py`: DimeNet++ global inference/evaluation script (single XYZ + batch prediction).
- `scripts/tsne_pipeline.py`: t-SNE workflow using inference output CSV.
- `configs/config.json`: Runtime configuration with numeric hyperparameters and desensitized paths.
- `docs/script_reference.md`: Script I/O and usage summary.
- `docs/dataset_format.md`: Expected CSV and XYZ formats for training and inference.
- `MCD_full_version_20260605.csv`: Public full-database release file for the complete MCD collection.
- `training_data_01.csv`: Public training dataset used in this study (7,749 structures; 2 columns: `Refcode`, `Label`; no header).

## Quick Start

Runtime prerequisite:
- `scripts/tsne_pipeline.py` currently requires CUDA-enabled PyTorch and an available NVIDIA GPU.
- CPU-only environments are not supported for this script in the current release.

1. Fill local path placeholders in `configs/config.json`.
2. Prepare training dataset files according to `docs/dataset_format.md`.
   - The repository includes `training_data_01.csv` at the repository root and a working copy at `data/labels/training_data_01.csv`.
   - The default working training label CSV is `data/labels/training_data_01.csv`.
   - `training_data_01.csv` contains 7,749 structures, uses no header, and includes two columns: `Refcode` and `Label`.
   - In this released training set, all macrocyclic and cage structures are uniformly labeled as `1`.
   - Global inference refcode CSV is manually provided, one column, no header.
3. Run extraction:

```bash
python scripts/csd_extractor.py
```

4. Open and run training notebook:

- `scripts/train_dimenet_schnet.ipynb`

5. Run global inference:

```bash
python scripts/dimenet_inference.py \
  --mode infer \
  --refcode-csv data/inference/refcodes_for_infer.csv \
  --xyz-dir data/extracted/CSD_Molecule \
  --out-csv outputs/inference/DNNetBatchPredictions.csv
```

6. Run t-SNE using the inference output file:

```bash
python scripts/tsne_pipeline.py
```

7. (Optional) Run labeled evaluation:

```bash
python scripts/dimenet_inference.py --mode eval --label-csv data/labels/training_data_01.csv --label-mode auto
```

## Current Repository Structure

```text
<repo-root>/
├─ LICENSE
├─ README.md
├─ RELEASE_PACKAGE_MANIFEST.md
├─ SECURITY.md
├─ MCD_full_version_20260605.csv
├─ training_data_01.csv
├─ requirements.txt
├─ configs/
│  └─ config.json
├─ data/
│  └─ labels/
│     └─ training_data_01.csv
├─ docs/
│  ├─ dataset_format.md
│  └─ script_reference.md
└─ scripts/
   ├─ csd_extractor.py
   ├─ train_dimenet_schnet.ipynb
   ├─ dimenet_inference.py
   └─ tsne_pipeline.py
```

## Release Scope

Included:
- workflow scripts under `scripts/`
- runtime config under `configs/config.json`
- core docs under `docs/`
- public full-database file `MCD_full_version_20260605.csv`
- public training dataset `training_data_01.csv`
- working training label CSV `data/labels/training_data_01.csv`

Not included:
- `workspace/`
- `outputs/`
- model weights (`*.pth`, `*.pt`, `*.ckpt`)
- cached feature arrays (`*.npy`, `*.npz`)

## Recommended Workspace Layout

Use one project root folder (for example `workspace/`) and keep scripts under `scripts/`.

```text
workspace/
├─ MCD_full_version_20260605.csv
├─ configs/
│  └─ config.json
├─ scripts/
│  ├─ csd_extractor.py
│  ├─ train_dimenet_schnet.ipynb
│  ├─ tsne_pipeline.py
│  └─ dimenet_inference.py
├─ docs/
│  ├─ script_reference.md
│  └─ dataset_format.md
├─ data/
│  ├─ labels/
│  │  └─ training_data_01.csv
│  ├─ inference/                     # optional staging area
│  │  └─ refcodes_for_infer.csv      # one column, no header
│  └─ extracted/
│     ├─ CSD_CIF/
│     ├─ CSD_Molecule/               # canonical XYZ source for train/infer/tsne
│     ├─ CSD_CIF_TEST/
│     └─ CSD_Molecule_TEST/
├─ models/
│  └─ best_DimeNetplus.pth
└─ outputs/
   ├─ training_run_<TIMESTAMP>/
   ├─ inference/
   │  └─ DNNetBatchPredictions.csv   # no header, 2 columns: Refcode, Predicted_Class
   ├─ model_weights/
   │  ├─ best_DimeNetplus.pth
   │  └─ best_SchNet.pth
   ├─ evaluation/
   │  ├─ classification_report.txt   # should include F1/Accuracy/AUC
   │  └─ predictions_test.csv
   ├─ X_full_features.npy
   ├─ y_predicted_class.npy
   ├─ X_full_features_pca50.npy
   ├─ tsne_master_inference_results.csv
   └─ Plot_DimeNet_Predictions_Full.png
```

### Config Mapping

- `paths.data_root` -> `workspace/data/extracted` or your chosen base path
- `paths.xyz_dir` -> `workspace/data/extracted/CSD_Molecule`
- `paths.label_csv` -> `workspace/data/labels/training_data_01.csv`
- `paths.inference_csv` -> `workspace/outputs/inference/DNNetBatchPredictions.csv`
- `paths.dimenet_weights` -> `workspace/models/best_DimeNetplus.pth`
- `paths.output_root` -> `workspace/outputs`
- `MCD_full_version_20260605.csv` can be used by downstream selection or split scripts

### Usage Notes

- Keep all runnable scripts under `workspace/scripts/`.
- Keep runtime config under `workspace/configs/config.json`.
- Keep machine-specific paths only in config; do not hardcode them into scripts.
- The training notebook reads label CSV + XYZ files; make sure those two paths are valid before training.
- Keep one canonical XYZ folder for training/inference/t-SNE: `workspace/data/extracted/CSD_Molecule`.
- Use `data/labels/training_data_01.csv` as the default working training label CSV.
- Keep the repository-root `training_data_01.csv` as a public release copy of the same dataset.
- Use a one-column, no-header refcode CSV as global inference input.
- Use the inference output file referenced by `paths.inference_csv` as the t-SNE input.
- Recommended t-SNE input file: `workspace/outputs/inference/DNNetBatchPredictions.csv`
- Expected t-SNE input format: no header, column 1 = `Refcode`, column 2 = `Predicted_Class`

## Notes

- `scripts/tsne_pipeline.py` consumes inference CSV and XYZ files, then performs feature extraction + PCA+t-SNE.
- DimeNet++ model parameters in `build_model(...)` are read from `configs/config.json` and should match checkpoint architecture.
- Keep all machine-specific paths in config only.

### Label Definition for Released Training Dataset

`training_data_01.csv` is a binary training dataset with no header and two columns: `Refcode`, `Label`.

`MCD_full_version_20260605.csv` is the released full-database file for the complete MCD collection. When label annotations are present in this file, they follow the full-database convention: `0` = non-target, `1` = macrocycle, `2` = cage.

Released label definition:
- 0: negative
- 1: positive

In this released dataset, all macrocyclic and cage structures are uniformly assigned the label `1`.

Dataset summary:
- total structures: 7,749
- label 0: 3,858
- label 1: 3,891

For evaluation with this file, `scripts/dimenet_inference.py` should use `--label-mode auto` (or `zero_one`).

## Privacy

Use placeholders for public examples (`<DATA_ROOT>`, `<MODEL_ROOT>`, `<OUTPUT_ROOT>`, etc.) and keep private local paths in config only.

## Data and License Notice

- Code in this repository is distributed under the project `LICENSE`.
- Data access and usage involving CSD/CCDC resources must comply with their own license terms.
- `MCD_full_version_20260605.csv` is provided for workflow indexing/alignment in this repository context; any redistribution or external reuse must be checked against upstream data and institutional policy requirements.
