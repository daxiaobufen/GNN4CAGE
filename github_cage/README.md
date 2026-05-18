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
- `mcd_master_index.csv`: Master database list (final index table of all records).

## Quick Start

Runtime prerequisite:
- `scripts/tsne_pipeline.py` currently requires CUDA-enabled PyTorch and an available NVIDIA GPU.
- CPU-only environments are not supported for this script in the current release.

1. Fill local path placeholders in `configs/config.json`.
2. Prepare training dataset files according to `docs/dataset_format.md`.
   - Training label CSV is manually curated and provided by users.
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
python scripts/dimenet_inference.py --mode eval --label-mode mcd_012
```

## Current Repository Structure

```text
<repo-root>/
├─ mcd_master_index.csv
├─ README.md
├─ README_zh.md
├─ RELEASE_PACKAGE_MANIFEST.md
├─ requirements.txt
├─ configs/
│  └─ config.json
├─ docs/
│  ├─ dataset_format.md
│  ├─ dataset_format_zh.md
│  ├─ script_reference.md
│  ├─ script_reference_zh.md
│  └─ release_checklist.md
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
- master list `mcd_master_index.csv`

Not included:
- `workspace/`
- `outputs/`
- model weights (`*.pth`, `*.pt`, `*.ckpt`)
- cached feature arrays (`*.npy`, `*.npz`)

## Recommended Workspace Layout

Use one project root folder (for example `workspace/`) and keep scripts under `scripts/`.

```text
workspace/
├─ mcd_master_index.csv
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
│  │  └─ labels_master_mcd.csv
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
- `paths.label_csv` -> `workspace/data/labels/labels_master_mcd.csv`
- `paths.inference_csv` -> `workspace/outputs/inference/DNNetBatchPredictions.csv`
- `paths.dimenet_weights` -> `workspace/models/best_DimeNetplus.pth`
- `paths.output_root` -> `workspace/outputs`
- `mcd_master_index.csv` can be used by downstream selection or split scripts

### Usage Notes

- Keep all runnable scripts under `workspace/scripts/`.
- Keep runtime config under `workspace/configs/config.json`.
- Keep machine-specific paths only in config; do not hardcode them into scripts.
- The training notebook reads label CSV + XYZ files; make sure those two paths are valid before training.
- Keep one canonical XYZ folder for training/inference/t-SNE: `workspace/data/extracted/CSD_Molecule`.
- Keep the training label CSV as a manually curated file.
- Use a one-column, no-header refcode CSV as global inference input.
- Use the inference output file referenced by `paths.inference_csv` as the t-SNE input.
- Recommended t-SNE input file: `workspace/outputs/inference/DNNetBatchPredictions.csv`
- Expected t-SNE input format: no header, column 1 = `Refcode`, column 2 = `Predicted_Class`

## Notes

- `scripts/tsne_pipeline.py` consumes inference CSV and XYZ files, then performs feature extraction + PCA+t-SNE.
- DimeNet++ model parameters in `build_model(...)` are read from `configs/config.json` and should match checkpoint architecture.
- Keep all machine-specific paths in config only.

### Label Mapping for Binary Training

Raw CSV labels:
- 0: negative
- 1: macrocycle
- 2: porous cage

Binary training mapping:
- 0 -> 0
- 1 -> 1
- 2 -> 1

This same mapping is used in:
- `scripts/dimenet_inference.py` (`--label-mode mcd_012`)
- `scripts/tsne_pipeline.py` for `Predicted_Class` normalization

## Privacy

Use placeholders for public examples (`<DATA_ROOT>`, `<MODEL_ROOT>`, `<OUTPUT_ROOT>`, etc.) and keep private local paths in config only.

## Data and License Notice

- Code in this repository is distributed under the project `LICENSE`.
- Data access and usage involving CSD/CCDC resources must comply with their own license terms.
- `mcd_master_index.csv` is provided for workflow indexing/alignment in this repository context; any redistribution or external reuse must be checked against upstream data and institutional policy requirements.
