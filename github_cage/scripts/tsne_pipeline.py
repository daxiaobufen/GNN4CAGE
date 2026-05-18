"""
Script purpose:
- Run full DimeNet++ inference feature extraction from XYZ files.
- Perform PCA + t-SNE projection and export CSV/plot artifacts.
"""

import os
import time
import traceback
from collections import Counter
import json

# Windows + sklearn PCA stability guard for huge matrices
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase.data import atomic_numbers
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from torch_geometric.data import Batch, Data
from torch_geometric.nn import DimeNetPlusPlus


def load_runtime_config():
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.json"))
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_runtime_config()

# =========================
# Global directives
# =========================
EXPECTED_PYTHON = CFG.get("paths", {}).get("python_executable", "<PYTHON_EXECUTABLE>")

INFERENCE_CSV = CFG.get("paths", {}).get("inference_csv", "<INFERENCE_CSV>")
STRUCTURE_DIR = CFG.get("paths", {}).get("xyz_dir", "<XYZ_DIR>")
MODEL_WEIGHTS = CFG.get("paths", {}).get("dimenet_weights", "<DIMENET_WEIGHTS>")

OUT_ROOT = CFG.get("paths", {}).get("output_root", "<OUTPUT_ROOT>")
FEATURES_OUT = os.path.join(OUT_ROOT, "X_full_features.npy")
PRED_OUT = os.path.join(OUT_ROOT, "y_predicted_class.npy")
PCA_50D_OUT = os.path.join(OUT_ROOT, "X_full_features_pca50.npy")
TSNE_MASTER_OUT = os.path.join(OUT_ROOT, "tsne_master_inference_results.csv")
PLOT_OUT = os.path.join(OUT_ROOT, "Plot_DimeNet_Predictions_Full.png")
TMP_FEATURES_MEMMAP = FEATURES_OUT + ".tmp.memmap"
TMP_PREDS_MEMMAP = PRED_OUT + ".tmp.memmap"
EXTRACT_CHECKPOINT = os.path.join(OUT_ROOT, "extract_checkpoint.json")

RANDOM_SEED = int(CFG.get("tsne", {}).get("seed", 42))
FEATURE_DIM = int(CFG.get("tsne", {}).get("feature_dim", 256))
BATCH_SIZE = int(CFG.get("tsne", {}).get("batch_size", 24))
PERPLEXITY = float(CFG.get("tsne", {}).get("perplexity", 80))
CHECKPOINT_EVERY_ROWS = int(CFG.get("tsne", {}).get("checkpoint_every_rows", 2000))

# Use cache for heavy stages if output already exists
FORCE_REEXTRACT = False
FORCE_REREDUCE = False


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def check_environment() -> torch.device:
    log("Phase 1/5 - Environment setup")
    log(f"Current Python executable: {os.path.abspath(os.sys.executable)}")
    log(f"Expected Python executable: {EXPECTED_PYTHON}")
    if EXPECTED_PYTHON != "<PYTHON_EXECUTABLE>":
        if os.path.abspath(os.sys.executable).lower() != os.path.abspath(EXPECTED_PYTHON).lower():
            log("WARNING: interpreter is not the required absolute path. Continue anyway.")

    if not os.path.exists(INFERENCE_CSV):
        raise FileNotFoundError(f"Inference CSV not found: {INFERENCE_CSV}")
    if not os.path.isdir(STRUCTURE_DIR):
        raise FileNotFoundError(f"Structure dir not found: {STRUCTURE_DIR}")
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")

    log("Checking CUDA availability...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU inference is mandatory.")
    device = torch.device("cuda")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"CUDA device count: {torch.cuda.device_count()}")
    log(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    backend_probe = {}
    for name, mod in [
        ("cuml", "cuml"),
        ("tsnecuda", "tsnecuda"),
        ("torchdr", "torchdr"),
        ("tsne_torch", "tsne_torch"),
        ("openTSNE", "openTSNE"),
    ]:
        try:
            __import__(mod)
            backend_probe[name] = True
        except Exception:
            backend_probe[name] = False
    log(f"t-SNE backend availability: {backend_probe}")
    return device


def find_column(columns, preferred_name):
    low_map = {str(c).strip().lower(): c for c in columns}
    key = preferred_name.strip().lower()
    if key not in low_map:
        raise KeyError(f"Column '{preferred_name}' not found. Existing: {list(columns)}")
    return low_map[key]


def normalize_mcd_label(raw_label: int) -> int:
    # Unified MCD-minus convention: 0->0, 1/2->1
    if raw_label == 0:
        return 0
    if raw_label in (1, 2):
        return 1
    raise ValueError(f"Unexpected label value: {raw_label}")


def build_model(device: torch.device) -> DimeNetPlusPlus:
    model_cfg = CFG.get("dimenet_model", {})
    model = DimeNetPlusPlus(
        hidden_channels=int(model_cfg["hidden_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        num_blocks=int(model_cfg["num_blocks"]),
        int_emb_size=int(model_cfg["int_emb_size"]),
        basis_emb_size=int(model_cfg["basis_emb_size"]),
        out_emb_channels=int(model_cfg["out_emb_channels"]),
        num_spherical=int(model_cfg["num_spherical"]),
        num_radial=int(model_cfg["num_radial"]),
        cutoff=float(model_cfg["cutoff"]),
        envelope_exponent=int(model_cfg["envelope_exponent"]),
        num_before_skip=int(model_cfg["num_before_skip"]),
        num_after_skip=int(model_cfg["num_after_skip"]),
        num_output_layers=int(model_cfg["num_output_layers"]),
    ).to(device)
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


class MultiScaleFeatureHook:
    """
    Capture model.output_blocks[i].lin forward_pre_hook inputs and do per-graph sum pooling.
    Final feature = sum across output blocks (multi-scale), still 256D.
    """

    def __init__(self, model: DimeNetPlusPlus):
        self.model = model
        self._batch_index = None
        self._num_graphs = 0
        self._collected = []
        self._handles = []
        for block in self.model.output_blocks:
            handle = block.lin.register_forward_pre_hook(self._pre_hook)
            self._handles.append(handle)

    def _pre_hook(self, module, inputs):
        x = inputs[0]
        if self._batch_index is None:
            return None
        pooled = torch.zeros(
            (self._num_graphs, x.size(-1)),
            dtype=x.dtype,
            device=x.device,
        )
        pooled.index_add_(0, self._batch_index, x)
        self._collected.append(pooled)
        return None

    @torch.no_grad()
    def extract_batch(self, z, pos, batch):
        self._collected = []
        self._batch_index = batch
        self._num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        _ = self.model(z, pos, batch)
        self._batch_index = None
        if not self._collected:
            raise RuntimeError("No hook features were collected from output_blocks[i].lin.")
        stacked = torch.stack(self._collected, dim=0)
        return stacked.sum(dim=0)

    def close(self):
        for h in self._handles:
            h.remove()


def parse_xyz(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if not first:
            raise ValueError("Empty xyz file")
        n_atoms = int(first)
        _ = f.readline()
        z_list = []
        pos_list = []
        for _i in range(n_atoms):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 4:
                raise ValueError("Malformed xyz atom line")
            symbol = parts[0]
            if symbol not in atomic_numbers:
                raise ValueError(f"Unknown element symbol: {symbol}")
            z_list.append(atomic_numbers[symbol])
            pos_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
    z = np.asarray(z_list, dtype=np.int64)
    pos = np.asarray(pos_list, dtype=np.float32)
    return z, pos


def geometry_guard(pos: np.ndarray):
    if pos.shape[0] < 4:
        return False, "atoms_lt_4"
    tree = cKDTree(pos)
    dists, _ = tree.query(pos, k=2)
    nn = dists[:, 1]
    if float(np.min(nn)) < 0.6:
        return False, "distance_lt_0.6"
    if np.any(nn >= 5.0):
        return False, "isolated_atom_ge_5.0"
    return True, "ok"


def recursive_infer(
    model: DimeNetPlusPlus,
    hook: MultiScaleFeatureHook,
    data_list,
    device: torch.device,
):
    if len(data_list) == 0:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    try:
        pyg_batch = Batch.from_data_list(data_list).to(device)
        feats = hook.extract_batch(
            z=pyg_batch.z,
            pos=pyg_batch.pos,
            batch=pyg_batch.batch,
        )
        return feats.detach().cpu().numpy().astype(np.float32, copy=False)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg and len(data_list) > 1:
            torch.cuda.empty_cache()
            mid = len(data_list) // 2
            left = recursive_infer(model, hook, data_list[:mid], device)
            right = recursive_infer(model, hook, data_list[mid:], device)
            return np.concatenate([left, right], axis=0)
        raise


def extract_full_features(device: torch.device):
    if (not FORCE_REEXTRACT) and os.path.exists(FEATURES_OUT) and os.path.exists(PRED_OUT):
        log("Phase 2/5 - Feature files already exist, skipping extraction.")
        X = np.load(FEATURES_OUT, mmap_mode="r")
        y = np.load(PRED_OUT, mmap_mode="r")
        return int(X.shape[0]), int(y.shape[0]), Counter(y.tolist()), Counter()

    log("Phase 2/5 - Massive feature extraction (GPU, no sampling)")
    # Accept both formats:
    # 1) header CSV with columns Refcode, Predicted_Class
    # 2) no-header 2-column CSV: col0=Refcode, col1=Predicted_Class
    try:
        df = pd.read_csv(INFERENCE_CSV, low_memory=False)
        ref_col = find_column(df.columns, "Refcode")
        pred_col = find_column(df.columns, "Predicted_Class")
    except Exception:
        df = pd.read_csv(
            INFERENCE_CSV,
            header=None,
            usecols=[0, 1],
            names=["Refcode", "Predicted_Class"],
            low_memory=False,
        )
        ref_col = "Refcode"
        pred_col = "Predicted_Class"

    refcodes = df[ref_col].astype(str).str.strip().tolist()
    preds_raw = pd.to_numeric(df[pred_col], errors="coerce").tolist()
    total_rows = len(df)
    log(f"Total CSV rows: {total_rows}")

    model = build_model(device)
    hook = MultiScaleFeatureHook(model)

    start_idx = 0
    valid_count = 0
    invalid_reasons = Counter()
    class_counter = Counter()
    if (
        (not FORCE_REEXTRACT)
        and os.path.exists(EXTRACT_CHECKPOINT)
        and os.path.exists(TMP_FEATURES_MEMMAP)
        and os.path.exists(TMP_PREDS_MEMMAP)
    ):
        with open(EXTRACT_CHECKPOINT, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if int(ckpt.get("total_rows", -1)) == total_rows:
            start_idx = int(ckpt.get("next_idx", 0))
            valid_count = int(ckpt.get("valid_count", 0))
            class_counter = Counter({int(k): int(v) for k, v in ckpt.get("class_counter", {}).items()})
            invalid_reasons = Counter({str(k): int(v) for k, v in ckpt.get("invalid_reasons", {}).items()})
            log(
                f"Resuming extraction from checkpoint: next_idx={start_idx}, "
                f"valid_count={valid_count}"
            )
            feat_mem = np.memmap(
                TMP_FEATURES_MEMMAP, mode="r+", dtype=np.float32, shape=(total_rows, FEATURE_DIM)
            )
            pred_mem = np.memmap(
                TMP_PREDS_MEMMAP, mode="r+", dtype=np.int16, shape=(total_rows,)
            )
        else:
            log("Checkpoint rows mismatch. Restarting extraction from scratch.")
            feat_mem = np.memmap(
                TMP_FEATURES_MEMMAP, mode="w+", dtype=np.float32, shape=(total_rows, FEATURE_DIM)
            )
            pred_mem = np.memmap(
                TMP_PREDS_MEMMAP, mode="w+", dtype=np.int16, shape=(total_rows,)
            )
    else:
        feat_mem = np.memmap(
            TMP_FEATURES_MEMMAP, mode="w+", dtype=np.float32, shape=(total_rows, FEATURE_DIM)
        )
        pred_mem = np.memmap(
            TMP_PREDS_MEMMAP, mode="w+", dtype=np.int16, shape=(total_rows,)
        )

    pending_graphs = []
    pending_preds = []
    start = time.time()

    def save_checkpoint(next_idx: int):
        payload = {
            "next_idx": int(next_idx),
            "valid_count": int(valid_count),
            "total_rows": int(total_rows),
            "class_counter": dict(class_counter),
            "invalid_reasons": dict(invalid_reasons),
        }
        with open(EXTRACT_CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def flush_pending():
        nonlocal valid_count
        if not pending_graphs:
            return
        feats = recursive_infer(model, hook, pending_graphs, device)
        n = feats.shape[0]
        feat_mem[valid_count : valid_count + n] = feats
        pred_mem[valid_count : valid_count + n] = np.asarray(pending_preds, dtype=np.int16)
        for p in pending_preds:
            class_counter[int(p)] += 1
        valid_count += n
        pending_graphs.clear()
        pending_preds.clear()

    for idx0 in range(start_idx, total_rows):
        idx = idx0 + 1
        refcode = refcodes[idx0]
        pred_val = preds_raw[idx0]

        if idx % 5000 == 0:
            elapsed = time.time() - start
            rows_done = idx - start_idx
            speed = rows_done / elapsed if elapsed > 0 else 0.0
            log(
                f"Processed rows: {idx}/{total_rows}, "
                f"valid: {valid_count}, speed: {speed:.2f} rows/s"
            )
        if idx % CHECKPOINT_EVERY_ROWS == 0:
            flush_pending()
            feat_mem.flush()
            pred_mem.flush()
            save_checkpoint(next_idx=idx0 + 1)
            log(f"Checkpoint saved at row {idx}/{total_rows}")

        if refcode == "" or refcode.lower() == "nan":
            invalid_reasons["empty_refcode"] += 1
            continue
        if pd.isna(pred_val):
            invalid_reasons["invalid_predicted_class"] += 1
            continue

        try:
            pred_int = normalize_mcd_label(int(pred_val))
        except Exception:
            invalid_reasons["invalid_predicted_class_value"] += 1
            continue
        xyz_path = os.path.join(STRUCTURE_DIR, f"{refcode}.xyz")
        if not os.path.exists(xyz_path):
            invalid_reasons["missing_xyz"] += 1
            continue

        try:
            z_np, pos_np = parse_xyz(xyz_path)
            ok, reason = geometry_guard(pos_np)
            if not ok:
                invalid_reasons[reason] += 1
                continue

            graph = Data(
                z=torch.from_numpy(z_np).long(),
                pos=torch.from_numpy(pos_np).float(),
            )
            pending_graphs.append(graph)
            pending_preds.append(pred_int)
            if len(pending_graphs) >= BATCH_SIZE:
                flush_pending()
        except Exception as e:
            invalid_reasons[f"parse_or_infer_error:{type(e).__name__}"] += 1
            continue

    flush_pending()
    feat_mem.flush()
    pred_mem.flush()
    save_checkpoint(next_idx=total_rows)
    hook.close()

    np.save(FEATURES_OUT, feat_mem[:valid_count])
    np.save(PRED_OUT, pred_mem[:valid_count])
    del feat_mem
    del pred_mem
    if os.path.exists(TMP_FEATURES_MEMMAP):
        os.remove(TMP_FEATURES_MEMMAP)
    if os.path.exists(TMP_PREDS_MEMMAP):
        os.remove(TMP_PREDS_MEMMAP)
    if os.path.exists(EXTRACT_CHECKPOINT):
        os.remove(EXTRACT_CHECKPOINT)

    log(f"Saved features to: {FEATURES_OUT}")
    log(f"Saved predicted classes to: {PRED_OUT}")
    return valid_count, total_rows, class_counter, invalid_reasons


def run_tsne_with_fallback(X_50d: np.ndarray):
    errors = []

    # 1) cuML (GPU)
    try:
        from cuml.manifold import TSNE as CuMLTSNE  # type: ignore

        log("Trying GPU t-SNE backend: cuML")
        tsne = CuMLTSNE(
            n_components=2,
            perplexity=PERPLEXITY,
            random_state=RANDOM_SEED,
            init="random",
        )
        emb = tsne.fit_transform(X_50d)
        return np.asarray(emb), "cuml"
    except Exception as e:
        errors.append(f"cuml failed: {e}")

    # 2) tsnecuda (GPU)
    try:
        from tsnecuda import TSNE as TsneCudaTSNE  # type: ignore

        log("Trying GPU t-SNE backend: tsnecuda")
        tsne = TsneCudaTSNE(
            n_components=2,
            perplexity=PERPLEXITY,
            random_seed=RANDOM_SEED,
        )
        emb = tsne.fit_transform(X_50d.astype(np.float32, copy=False))
        return np.asarray(emb), "tsnecuda"
    except Exception as e:
        errors.append(f"tsnecuda failed: {e}")

    # 3) torchdr (GPU)
    try:
        from torchdr import TSNE as TorchDRTSNE  # type: ignore

        log("Trying GPU t-SNE backend: torchdr")
        x_gpu = torch.as_tensor(X_50d, dtype=torch.float32, device="cuda")
        tsne = TorchDRTSNE(
            perplexity=PERPLEXITY,
            n_components=2,
            device="cuda",
            random_state=RANDOM_SEED,
            max_iter=1000,
            verbose=True,
        )
        emb = tsne.fit_transform(x_gpu).detach().cpu().numpy()
        return emb, "torchdr"
    except Exception as e:
        errors.append(f"torchdr failed: {e}")

    # 4) tsne_torch (GPU)
    try:
        from tsne_torch import TorchTSNE  # type: ignore

        log("Trying GPU t-SNE backend: tsne_torch")
        x_gpu = torch.as_tensor(X_50d, dtype=torch.float32, device="cuda")
        tsne = TorchTSNE(
            n_components=2,
            perplexity=float(PERPLEXITY),
            n_iter=1000,
            verbose=True,
        )
        emb = tsne.fit_transform(x_gpu)
        return np.asarray(emb), "tsne_torch"
    except Exception as e:
        errors.append(f"tsne_torch failed: {e}")

    # 5) fallback openTSNE CPU
    try:
        from openTSNE import TSNE as OpenTSNE  # type: ignore

        log("GPU t-SNE backends unavailable. Falling back to openTSNE (CPU, n_jobs=-1).")
        tsne = OpenTSNE(
            n_components=2,
            perplexity=PERPLEXITY,
            initialization="pca",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            negative_gradient_method="fft",
        )
        emb = tsne.fit(X_50d.astype(np.float32, copy=False))
        return np.asarray(emb), "openTSNE_cpu_fallback"
    except Exception as e:
        errors.append(f"openTSNE failed: {e}")

    raise RuntimeError("All t-SNE backends failed:\n" + "\n".join(errors))


def reduce_and_save_tsne():
    if (not FORCE_REREDUCE) and os.path.exists(TSNE_MASTER_OUT):
        log("Phase 3/5 - Existing t-SNE master file found, skipping reduction.")
        tsne_df = pd.read_csv(TSNE_MASTER_OUT)
        return tsne_df, "cached"

    log("Phase 3/5 - GPU-accelerated dimensionality reduction")
    if not os.path.exists(FEATURES_OUT) or not os.path.exists(PRED_OUT):
        raise FileNotFoundError("Required feature files are missing.")

    X = np.load(FEATURES_OUT)
    y = np.load(PRED_OUT)
    if X.ndim != 2:
        raise ValueError(f"X shape invalid: {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y row mismatch: {X.shape[0]} vs {y.shape[0]}")

    log(f"Loaded feature matrix shape: {X.shape}")
    pca = PCA(
        n_components=min(50, X.shape[1]),
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )
    X_50d = pca.fit_transform(X).astype(np.float32, copy=False)
    log(f"PCA output shape: {X_50d.shape}")

    emb_2d, backend = run_tsne_with_fallback(X_50d)
    log(f"t-SNE backend used: {backend}")

    tsne_df = pd.DataFrame(
        {
            "tsne_1": emb_2d[:, 0],
            "tsne_2": emb_2d[:, 1],
            "y_predicted_class": y.astype(np.int16),
        }
    )
    tsne_df.to_csv(TSNE_MASTER_OUT, index=False)
    log(f"Saved t-SNE master CSV to: {TSNE_MASTER_OUT}")
    return tsne_df, backend


def plot_predictions(tsne_df: pd.DataFrame):
    log("Phase 4/5 - Model prediction visualization")
    required_cols = {"tsne_1", "tsne_2", "y_predicted_class"}
    if not required_cols.issubset(set(tsne_df.columns)):
        raise ValueError(f"Missing required columns in tsne df: {required_cols}")

    neg_df = tsne_df[tsne_df["y_predicted_class"] == 0]
    pos_df = tsne_df[tsne_df["y_predicted_class"] == 1]

    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    ax.scatter(
        neg_df["tsne_1"],
        neg_df["tsne_2"],
        c="darkgray",
        s=10,
        alpha=0.3,
        marker="o",
        zorder=1,
        label="Predicted Negative (Class 0)",
    )
    ax.scatter(
        pos_df["tsne_1"],
        pos_df["tsne_2"],
        c="#1F77B4",
        s=40,
        alpha=0.85,
        marker="o",
        zorder=2,
        label="Predicted Positive (Class 1)",
    )
    ax.set_title(f"DimeNet++ Feature Space: Model Prediction Distribution (Perplexity = {PERPLEXITY})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved plot to: {PLOT_OUT}")
    return int(neg_df.shape[0]), int(pos_df.shape[0])


def run_pipeline():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = check_environment()
    valid_count, total_rows, class_counter, invalid_reasons = extract_full_features(device)
    tsne_df, backend = reduce_and_save_tsne()
    neg_count, pos_count = plot_predictions(tsne_df)

    log("Phase 5/5 - Audit report")
    log(f"Successfully processed valid .xyz files: {valid_count} / {total_rows}")
    log(f"Predicted Negative (Class 0) plotted: {neg_count}")
    log(f"Predicted Positive (Class 1) plotted: {pos_count}")
    log(f"t-SNE backend summary: {backend}")
    log(f"Image generation confirmed: {PLOT_OUT}")
    if class_counter:
        log(f"Valid extracted class distribution: {dict(class_counter)}")
    if invalid_reasons:
        log(f"Dropped/invalid reason counts: {dict(invalid_reasons)}")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        log("Pipeline failed with exception:")
        log(str(exc))
        log(traceback.format_exc())
        raise
