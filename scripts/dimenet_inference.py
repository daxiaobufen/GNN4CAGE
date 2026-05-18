import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from ase.data import atomic_numbers
from torch_geometric.data import Data
from torch_geometric.nn import DimeNetPlusPlus


DEFAULT_WEIGHTS = ""


def load_config() -> Dict:
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "configs", "config.json")
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_xyz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if not first:
            raise ValueError("Empty xyz file")
        n_atoms = int(first)
        _ = f.readline()
        z_list: List[int] = []
        pos_list: List[List[float]] = []
        for _ in range(n_atoms):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 4:
                raise ValueError("Malformed xyz atom line")
            symbol = parts[0]
            # Isotope symbols are mapped to the same atomic number as hydrogen.
            if symbol in {"D", "T"}:
                symbol = "H"
            if symbol not in atomic_numbers:
                raise ValueError(f"Unknown element symbol: {symbol}")
            z_list.append(atomic_numbers[symbol])
            pos_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not z_list:
        raise ValueError("No valid atoms parsed from xyz")
    z = np.asarray(z_list, dtype=np.int64)
    pos = np.asarray(pos_list, dtype=np.float32)
    return z, pos


def build_model(cfg: Dict, weights_path: str, device: torch.device) -> DimeNetPlusPlus:
    model_cfg = cfg["dimenet_model"]
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
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_xyz(model: DimeNetPlusPlus, device: torch.device, xyz_path: str) -> Dict:
    z_np, pos_np = parse_xyz(xyz_path)
    z = torch.from_numpy(z_np).long().to(device)
    pos = torch.from_numpy(pos_np).float().to(device)
    batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
    logits = model(z=z, pos=pos, batch=batch).view(-1)
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    pred = int(torch.argmax(logits).item())
    return {
        "xyz_path": xyz_path,
        "n_atoms": int(z_np.shape[0]),
        "pred_class": pred,
        "prob_class_0": float(probs[0]),
        "prob_class_1": float(probs[1]),
        "logit_class_0": float(logits[0].item()),
        "logit_class_1": float(logits[1].item()),
    }


def infer_label_mode_auto(labels: pd.Series) -> str:
    uniq = sorted(set(int(v) for v in labels.dropna().tolist()))
    if set(uniq).issubset({0, 1}):
        return "zero_one"
    if set(uniq).issubset({1, 2}):
        return "minus_one"
    return "guard_non_binary_to_one"


def normalize_label(raw_label: int, mode: str) -> int:
    if mode == "mcd_012":
        if raw_label == 0:
            return 0
        if raw_label in (1, 2):
            return 1
        raise ValueError(f"Expected MCD label in {{0,1,2}}, got: {raw_label}")
    if mode == "zero_one":
        if raw_label in (0, 1):
            return int(raw_label)
        raise ValueError(f"Expected 0/1 label, got: {raw_label}")
    if mode == "minus_one":
        if raw_label in (1, 2):
            return int(raw_label - 1)
        if raw_label in (0, 1):
            return int(raw_label)
        raise ValueError(f"Expected 1/2 (or 0/1) label, got: {raw_label}")
    if mode == "guard_non_binary_to_one":
        return int(raw_label) if raw_label in (0, 1) else 1
    raise ValueError(f"Unknown label mode: {mode}")


def evaluate_batch(
    model: DimeNetPlusPlus,
    device: torch.device,
    label_csv: str,
    xyz_dir: str,
    out_csv: str,
    label_mode: str,
) -> Dict:
    df = pd.read_csv(label_csv, header=None)
    if df.shape[1] < 2:
        raise ValueError("Label CSV must have at least two columns: Refcode, Label")
    df = df.iloc[:, :2].copy()
    df.columns = ["Refcode", "Label"]

    if label_mode == "auto":
        selected_mode = infer_label_mode_auto(df["Label"])
    else:
        selected_mode = label_mode

    rows = []
    skipped = []
    correct = 0
    total_eval = 0

    for _, row in df.iterrows():
        refcode = str(row["Refcode"]).strip()
        try:
            raw_label = int(row["Label"])
        except Exception:
            skipped.append({"Refcode": refcode, "reason": "invalid_label"})
            continue
        xyz_path = os.path.join(xyz_dir, f"{refcode}.xyz")
        if not os.path.exists(xyz_path):
            skipped.append({"Refcode": refcode, "reason": "missing_xyz"})
            continue
        try:
            pred = predict_xyz(model, device, xyz_path)
            y_true = normalize_label(raw_label, selected_mode)
            y_pred = int(pred["pred_class"])
            is_correct = int(y_true == y_pred)
            correct += is_correct
            total_eval += 1
            rows.append(
                {
                    "Refcode": refcode,
                    "raw_label": raw_label,
                    "mapped_label": y_true,
                    "pred_class": y_pred,
                    "correct": is_correct,
                    "prob_class_0": pred["prob_class_0"],
                    "prob_class_1": pred["prob_class_1"],
                    "n_atoms": pred["n_atoms"],
                    "xyz_path": xyz_path,
                }
            )
        except Exception as exc:
            skipped.append({"Refcode": refcode, "reason": f"infer_error:{type(exc).__name__}"})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    accuracy = (correct / total_eval) if total_eval > 0 else 0.0
    return {
        "label_mode_used": selected_mode,
        "total_rows_in_label_csv": int(df.shape[0]),
        "evaluated_rows": int(total_eval),
        "correct_rows": int(correct),
        "accuracy": float(accuracy),
        "skipped_rows": int(len(skipped)),
        "skipped_detail": skipped,
        "predictions_csv": out_csv,
    }


def infer_batch_from_refcodes(
    model: DimeNetPlusPlus,
    device: torch.device,
    refcode_csv: str,
    xyz_dir: str,
    out_csv_no_header: str,
    out_detailed_csv: str,
) -> Dict:
    # Global inference input contract:
    # - one column, no header
    # - each row is a refcode
    df = pd.read_csv(refcode_csv, header=None)
    if df.shape[1] < 1:
        raise ValueError("Refcode CSV must have at least one column.")
    refcodes = df.iloc[:, 0].astype(str).str.strip().tolist()

    simple_rows = []
    detailed_rows = []
    skipped = []

    for refcode in refcodes:
        if refcode == "" or refcode.lower() == "nan":
            skipped.append({"Refcode": refcode, "reason": "empty_refcode"})
            continue
        xyz_path = os.path.join(xyz_dir, f"{refcode}.xyz")
        if not os.path.exists(xyz_path):
            skipped.append({"Refcode": refcode, "reason": "missing_xyz"})
            continue
        try:
            pred = predict_xyz(model, device, xyz_path)
            y_pred = int(pred["pred_class"])
            # Output contract requested by user:
            # - no header
            # - column 1: refcode, column 2: predicted label
            simple_rows.append([refcode, y_pred])
            detailed_rows.append(
                {
                    "Refcode": refcode,
                    "pred_label": y_pred,
                    "prob_class_0": pred["prob_class_0"],
                    "prob_class_1": pred["prob_class_1"],
                    "n_atoms": pred["n_atoms"],
                    "xyz_path": xyz_path,
                }
            )
        except Exception as exc:
            skipped.append({"Refcode": refcode, "reason": f"infer_error:{type(exc).__name__}"})

    # Save no-header 2-column output for downstream t-SNE input.
    pd.DataFrame(simple_rows).to_csv(out_csv_no_header, index=False, header=False, encoding="utf-8")
    pd.DataFrame(detailed_rows).to_csv(out_detailed_csv, index=False, encoding="utf-8")

    return {
        "mode": "infer",
        "total_refcodes": int(len(refcodes)),
        "predicted_rows": int(len(simple_rows)),
        "skipped_rows": int(len(skipped)),
        "skipped_detail": skipped,
        "predictions_csv_no_header": out_csv_no_header,
        "predictions_csv_detailed": out_detailed_csv,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DimeNet++ single XYZ inference, global inference, and batch accuracy evaluation."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Path to best_DimeNetplus.pth. If empty, use paths.dimenet_weights in config.",
    )
    parser.add_argument("--xyz", type=str, default="", help="Run single-file inference on this XYZ")
    parser.add_argument(
        "--mode",
        type=str,
        default="infer",
        choices=["infer", "eval"],
        help="infer: no-label global inference from refcode list; eval: evaluate with labeled CSV.",
    )
    parser.add_argument(
        "--refcode-csv",
        type=str,
        default="",
        help="No-header one-column refcode CSV for global inference.",
    )
    parser.add_argument(
        "--label-csv",
        type=str,
        default="",
        help="Label CSV for batch evaluation (2 columns: Refcode, Label)",
    )
    parser.add_argument("--xyz-dir", type=str, default="", help="Directory containing XYZ files")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="auto",
        choices=["auto", "mcd_012", "zero_one", "minus_one", "guard_non_binary_to_one"],
        help="How to map label values to model classes 0/1.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Output CSV path. infer mode: no-header 2-column (Refcode,PredLabel).",
    )
    parser.add_argument(
        "--out-detailed-csv",
        type=str,
        default="",
        help="Detailed inference CSV with probabilities (infer mode only).",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="",
        help="Output JSON summary for batch evaluation",
    )
    args = parser.parse_args()

    cfg = load_config()
    default_label_csv = cfg["paths"]["label_csv"]
    default_xyz_dir = cfg["paths"]["xyz_dir"]
    default_out_root = cfg["paths"]["output_root"]
    default_inference_csv = cfg["paths"].get("inference_csv", "")

    label_csv = args.label_csv if args.label_csv else default_label_csv
    refcode_csv = args.refcode_csv
    xyz_dir = args.xyz_dir if args.xyz_dir else default_xyz_dir
    out_csv = (
        args.out_csv
        if args.out_csv
        else (default_inference_csv if default_inference_csv else os.path.join(default_out_root, "DNNetBatchPredictions.csv"))
    )
    out_detailed_csv = (
        args.out_detailed_csv
        if args.out_detailed_csv
        else os.path.join(default_out_root, "DNNetBatchPredictions_detailed.csv")
    )
    out_summary = (
        args.out_summary
        if args.out_summary
        else os.path.join(default_out_root, "dimenet_batch_eval_summary.json")
    )

    weights_path = args.weights if args.weights else cfg["paths"]["dimenet_weights"]
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] weights={weights_path}")
    model = build_model(cfg, weights_path, device)

    if args.xyz:
        if not os.path.exists(args.xyz):
            raise FileNotFoundError(f"XYZ not found: {args.xyz}")
        single = predict_xyz(model, device, args.xyz)
        print("[SINGLE]", json.dumps(single, ensure_ascii=False))

    if args.mode == "infer":
        if not refcode_csv:
            # Backward-compatible fallback: use label_csv first column as refcode list.
            refcode_csv = label_csv
        summary = infer_batch_from_refcodes(
            model=model,
            device=device,
            refcode_csv=refcode_csv,
            xyz_dir=xyz_dir,
            out_csv_no_header=out_csv,
            out_detailed_csv=out_detailed_csv,
        )
    else:
        summary = evaluate_batch(
            model=model,
            device=device,
            label_csv=label_csv,
            xyz_dir=xyz_dir,
            out_csv=out_csv,
            label_mode=args.label_mode,
        )
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[BATCH_SUMMARY]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
