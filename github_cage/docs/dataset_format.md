# Dataset Format (for scripts/train_dimenet_schnet.ipynb)

This document describes the dataset format expected by the training notebook.

## 1) Training label CSV format (manual)

The notebook reads CSV with `header=None`, then keeps the first two columns only:

- Column 1: `Refcode`
- Column 2: `Label`

Minimum valid CSV layout:

```csv
ABCD01,1
EFGH02,0
IJKL03,1
```

Rules:
- `Refcode` is trimmed as string and used to locate `.xyz` file.
- `Label` must be integer. Current supported raw values are `0/1/2`.
- Binary mapping rule used by scripts:
  - `0 -> 0`
  - `1 -> 1`
  - `2 -> 1`
- Other label values are treated as invalid.
- Extra columns are allowed but ignored by the current parser.
- This label CSV is prepared manually.

## 2) XYZ file format

For each `Refcode`, the loader expects:

- file path: `<XYZ_DIR>/<Refcode>.xyz`
- file content:
  - line 1: atom count
  - line 2: comment / title
  - line 3+ : `ElementSymbol x y z`

Example:

```text
3
ABCD01
C 0.0000 0.0000 0.0000
N 1.2000 0.0000 0.0000
H -0.5000 0.9000 0.0000
```

Rules:
- If the `.xyz` file does not exist, that sample is skipped.
- If atom line has fewer than 4 columns, that atom line is ignored.
- Only elements included in `ATOM_MAP` are accepted.
- In inference utility (`scripts/dimenet_inference.py`), isotope symbols `D`/`T` are normalized to `H`.
- If no valid atoms remain, that sample is skipped.

## 3) PyG Data object schema

Each valid sample is converted to one `torch_geometric.data.Data`:

- `z`: `LongTensor[num_atoms]` (atomic numbers)
- `pos`: `FloatTensor[num_atoms, 3]` (3D coordinates)
- `y`: `LongTensor[1]` (class label)
- `refcode`: string metadata field

## 4) Repository note

Prepare the label CSV and XYZ directory in the format above.
Keep private local paths in `configs/config.json`.

## 5) Global inference input CSV (`scripts/dimenet_inference.py --mode infer`)

Expected format:
- one column
- no header
- each row is one `Refcode`

Example:

```csv
ACUNIV
ADIZOG
ADUGAH
```

## 6) Inference output CSV and t-SNE input CSV (`scripts/tsne_pipeline.py`)

Recommended file:
- `outputs/inference/DNNetBatchPredictions.csv`

Format:
- no header
- column 1: `Refcode`
- column 2: `Predicted_Class` (0/1)

Example:

```csv
ACUNIV,1
ADIZOG,0
ADUGAH,1
```

`scripts/tsne_pipeline.py` accepts:
- header format (`Refcode`, `Predicted_Class`)
- no-header 2-column format (canonical format above)
