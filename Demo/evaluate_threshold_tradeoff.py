import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_THRESHOLDS = [0.6224134795519164, 0.55, 0.50, 0.45, 0.40, 0.35, 0.335, 0.30]
DEFAULT_TARGET_RECALLS = [0.90, 0.92, 0.95]


def parse_prediction_arg(value):
    if ":" in value and not (len(value) > 1 and value[1] == ":"):
        name, path = value.split(":", 1)
        return name, Path(path)
    path = Path(value)
    return clean_name(path), path


def clean_name(path):
    stem = Path(path).stem
    for suffix in ("_clean_predictions", "_predictions"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def discover_prediction_files(run_dir):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("*_clean_predictions.csv"))
    if not files:
        files = sorted(run_dir.glob("*_predictions.csv"))
    return [(clean_name(path), path) for path in files]


def calc_auc_values(y_true, y_prob):
    if len(set(y_true)) < 2:
        return None, None, None, None

    auc_roc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall_curve, precision_curve)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return float(auc_roc), float(auc_pr), eer, eer_threshold


def metrics_at_threshold(dataset, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    auc_roc, auc_pr, eer, eer_threshold = calc_auc_values(y_true, y_prob)

    return {
        "dataset": dataset,
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "real": int((y_true == 0).sum()),
        "fake": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "f2": float(f2),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "far_fp_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "miss_fn_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_for_target_recall(dataset, y_true, y_prob, target_recall):
    candidates = sorted(set(float(x) for x in y_prob), reverse=True)
    candidates.extend([0.0])

    best = None
    for threshold in candidates:
        row = metrics_at_threshold(dataset, y_true, y_prob, threshold)
        if row["recall"] >= target_recall:
            best = row
            break

    if best is None:
        best = metrics_at_threshold(dataset, y_true, y_prob, 0.0)

    best["target_recall"] = float(target_recall)
    return best


def fmt_pct(value):
    if value is None or pd.isna(value):
        return ""
    return f"{100.0 * float(value):.2f}%"


def fmt_num(value):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def write_report(out_file, threshold_df, target_df):
    lines = []
    lines.append("# Threshold Trade-off Report")
    lines.append("")
    lines.append(
        "Lowering the threshold predicts Fake more easily. This usually increases recall "
        "and reduces missed Fake images, but also increases false positives on Real images."
    )
    lines.append("")

    for dataset, group in threshold_df.groupby("dataset", sort=False):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Threshold | Acc | Precision | Recall | F1 | F2 | FAR/FP rate | FN/Miss rate | FP | FN |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in group.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt_num(row["threshold"]),
                        fmt_pct(row["accuracy"]),
                        fmt_pct(row["precision"]),
                        fmt_pct(row["recall"]),
                        fmt_pct(row["f1"]),
                        fmt_pct(row["f2"]),
                        fmt_pct(row["far_fp_rate"]),
                        fmt_pct(row["miss_fn_rate"]),
                        str(int(row["fp"])),
                        str(int(row["fn"])),
                    ]
                )
                + " |"
            )
        lines.append("")

    if not target_df.empty:
        lines.append("## Thresholds For Target Recall")
        lines.append("")
        lines.append("| Dataset | Target recall | Max threshold | Precision | Recall | F1 | FAR/FP rate | FN |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in target_df.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["dataset"]),
                        fmt_pct(row["target_recall"]),
                        fmt_num(row["threshold"]),
                        fmt_pct(row["precision"]),
                        fmt_pct(row["recall"]),
                        fmt_pct(row["f1"]),
                        fmt_pct(row["far_fp_rate"]),
                        str(int(row["fn"])),
                    ]
                )
                + " |"
            )
        lines.append("")

    out_file.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lower-threshold precision/recall trade-offs from prediction CSV files.")
    parser.add_argument("--run-dir", default="evaluation_runs/20260806_094118")
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Prediction CSV as NAME:path or path. If omitted, clean prediction files are discovered from --run-dir.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--target-recall", nargs="+", type=float, default=DEFAULT_TARGET_RECALLS)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "threshold_tradeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = [parse_prediction_arg(value) for value in args.prediction]
    if not prediction_files:
        prediction_files = discover_prediction_files(run_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction CSV files found in {run_dir}")

    threshold_rows = []
    target_rows = []
    for dataset, path in prediction_files:
        if not path.is_absolute():
            path = Path(path)
        df = pd.read_csv(path)
        required = {"label", "prob_fake"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        y_true = df["label"].to_numpy(dtype=int)
        y_prob = df["prob_fake"].to_numpy(dtype=float)

        for threshold in args.thresholds:
            threshold_rows.append(metrics_at_threshold(dataset, y_true, y_prob, threshold))
        for target_recall in args.target_recall:
            target_rows.append(threshold_for_target_recall(dataset, y_true, y_prob, target_recall))

    threshold_df = pd.DataFrame(threshold_rows)
    target_df = pd.DataFrame(target_rows)
    threshold_df.to_csv(out_dir / "threshold_tradeoff.csv", index=False)
    target_df.to_csv(out_dir / "target_recall_thresholds.csv", index=False)
    write_report(out_dir / "THRESHOLD_TRADEOFF_REPORT.md", threshold_df, target_df)

    print(f"Saved threshold trade-off report: {out_dir / 'THRESHOLD_TRADEOFF_REPORT.md'}")
    print(f"Saved threshold table: {out_dir / 'threshold_tradeoff.csv'}")
    print(f"Saved target recall table: {out_dir / 'target_recall_thresholds.csv'}")


if __name__ == "__main__":
    main()
