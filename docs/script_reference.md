# Script Reference

## scripts/csd_extractor.py
- Purpose: Extract valid structures from CSD and export CIF/XYZ files.
- Input: CSD database via CCDC Python API, plus `extract` and `paths` entries in `configs/config.json`.
- Output:
  - CIF directory (`CSD_CIF` or `CSD_CIF_TEST`)
  - XYZ directory (`CSD_Molecule` or `CSD_Molecule_TEST`)
  - Valid refcode CSV (`valid_refcodes.csv` or `valid_refcodes_TEST.csv`)
- Notes: Uses paths and extraction settings from `configs/config.json`.

## scripts/train_dimenet_schnet.ipynb
- Purpose: Main training workflow notebook for DimeNet++ and SchNet.
- Input:
  - Manually curated label CSV (`Refcode, Label` format)
  - XYZ directory (recommended canonical source: `data/extracted/CSD_Molecule`)
  - `train`, `dimenet_model`, `schnet_model`, `paths` in `configs/config.json`
- Output:
  - Model weights (e.g., best DimeNet++/SchNet checkpoints)
  - Test/evaluation artifacts (classification report, prediction files)
  - Step logs (CSV)
  - Training curves
- Notes: Loads model and training parameters from `configs/config.json`.

## scripts/dimenet_inference.py
- Purpose: DimeNet++ inference utility for single XYZ prediction, global inference, and labeled evaluation.
- Input:
  - `--mode infer`: no-header one-column refcode CSV
  - `--mode eval`: labeled CSV (`Refcode`, `Label`)
  - XYZ directory
  - DimeNet++ model weights
- Output:
  - `DNNetBatchPredictions.csv` (no header, 2 columns: `Refcode, Predicted_Class`)
  - `DNNetBatchPredictions_detailed.csv` (detailed probabilities/metadata)
  - optional evaluation summaries in JSON
- Notes:
  - `mcd_012` mapping: `0->0, 1/2->1`
  - supports isotope normalization `D/T -> H`

## scripts/tsne_pipeline.py
- Purpose: Feature extraction + PCA + t-SNE projection pipeline using inference output CSV.
- Input:
  - Inference CSV from `dimenet_inference.py` output (canonical: `DNNetBatchPredictions.csv`)
  - XYZ directory
  - DimeNet++ model weights
  - `paths`, `dimenet_model`, `tsne` sections in `configs/config.json`
- Output:
  - `X_full_features.npy`
  - `y_predicted_class.npy`
  - `X_full_features_pca50.npy`
  - `tsne_master_inference_results.csv`
  - `Plot_DimeNet_Predictions_Full.png`
- Notes:
  - Accepts both header and no-header 2-column inference CSV.
  - Uses unified MCD mapping for `Predicted_Class`: `0->0, 1/2->1`.

## configs/config.json
- Purpose: Central runtime config file.
- Includes:
  - `paths`: desensitized machine-specific paths
  - `dimenet_model`: model architecture parameters used by training and t-SNE inference
  - `schnet_model`: SchNet architecture parameters for training
  - `train`: training hyperparameters
  - `extract`: extraction mode controls
  - `tsne`: t-SNE pipeline hyperparameters

## docs/dataset_format.md
- Purpose: Dataset format expected by the training notebook.
- Covers:
  - CSV column format
  - XYZ file format
  - PyG Data object fields
