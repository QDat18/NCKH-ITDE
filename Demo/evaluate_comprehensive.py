import argparse
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_pytorch import get_transforms, make_frequency_tensor
from model_pytorch import DeepfakeEfficientNet


LABEL_NAMES = {0: "Real", 1: "Fake"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class MetricRow:
    dataset: str
    condition: str
    samples: int
    real: int
    fake: int
    threshold: float
    eer_threshold: Optional[float]
    accuracy: Optional[float]
    balanced_accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    specificity: Optional[float]
    f1: Optional[float]
    auc_roc: Optional[float]
    auc_pr: Optional[float]
    average_precision: Optional[float]
    eer: Optional[float]
    far: Optional[float]
    frr: Optional[float]
    mcc: Optional[float]
    tn: Optional[int]
    fp: Optional[int]
    fn: Optional[int]
    tp: Optional[int]
    elapsed_sec: float
    images_per_sec: Optional[float]
    ms_per_image: Optional[float]


class EvalDataset(Dataset):
    def __init__(self, split_file, transform=None, degradation=None, limit=None, seed=42, frequency_mode="none"):
        self.split_file = str(split_file)
        self.transform = transform
        self.degradation = degradation
        self.rng = np.random.default_rng(seed)
        self.frequency_mode = frequency_mode or "none"
        self.samples = read_split(split_file)
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = apply_degradation(img, self.degradation, self.rng)
        if self.transform:
            img = self.transform(img)
        if self.frequency_mode != "none":
            freq = make_frequency_tensor(img, mode=self.frequency_mode)
            return (img, freq), int(label), str(path)
        return img, int(label), str(path)


def read_split(split_file):
    samples = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                if "," in line:
                    path, label = line.rsplit(",", 1)
                else:
                    path, label = line.rsplit(None, 1)
            except ValueError as exc:
                raise ValueError(f"Bad split line {line_no} in {split_file}: {raw!r}") from exc
            samples.append((path.strip(), int(label.strip())))
    return samples


def write_split(samples, split_file):
    with open(split_file, "w", encoding="utf-8") as f:
        for path, label in samples:
            f.write(f"{path},{label}\n")


def infer_label(path):
    text = str(path).lower()
    parts = set(Path(path).parts)
    lower_parts = {p.lower() for p in parts}
    if "real" in lower_parts or "0_real" in text or "original" in text:
        return 0
    if "fake" in lower_parts or "1_fake" in text or "manipulated" in text:
        return 1
    if "real" in text and "fake" not in text:
        return 0
    if "fake" in text or "deepfake" in text:
        return 1
    return None


def create_split_from_dir(dataset_dir, out_file):
    samples = []
    for root, _, files in os.walk(dataset_dir):
        for file_name in files:
            if Path(file_name).suffix.lower() not in IMAGE_EXTS:
                continue
            path = Path(root) / file_name
            label = infer_label(path)
            if label is not None:
                samples.append((str(path), label))
    samples.sort()
    write_split(samples, out_file)
    return samples


def source_name(path):
    text = str(path)
    lower = text.lower()
    base = os.path.basename(text)
    if "FF_" in base:
        return "FaceForensics++"
    if "Celeb-" in base:
        return "Celeb-DF"
    if "dfdc" in lower or "deepfake-detection-challenge" in lower:
        return "DFDC"
    if "real and fake face" in lower or "ciplab" in lower:
        return "CIPLAB/Kaggle"
    return "Other"


def extract_ids(path):
    ids = set()
    base = os.path.basename(str(path))
    if "FF_" in base:
        parts = base.split("_")
        for part in parts:
            if part.isdigit() and len(part) >= 2:
                ids.add(f"FF_{part}")
    elif "Celeb-" in base:
        parts = base.split("_")
        for part in parts:
            if part.startswith("id") and part[2:].isdigit():
                ids.add(f"Celeb_{part}")
    return ids


def identity_leakage_report(train_split, *other_splits):
    if not train_split or not Path(train_split).exists():
        return {}
    train_ids = set()
    for path, _ in read_split(train_split):
        train_ids.update(extract_ids(path))

    report = {
        "train_split": str(train_split),
        "train_unique_ids": len(train_ids),
        "comparisons": [],
    }
    for split in other_splits:
        if not split or not Path(split).exists():
            continue
        ids = set()
        for path, _ in read_split(split):
            ids.update(extract_ids(path))
        leaked = train_ids.intersection(ids)
        denominator = len(ids) or 1
        report["comparisons"].append(
            {
                "split": str(split),
                "unique_ids": len(ids),
                "leaked_ids": len(leaked),
                "leakage_percent": 100.0 * len(leaked) / denominator,
                "leaked_examples": sorted(leaked)[:20],
            }
        )
    return report


def apply_degradation(img, degradation, rng):
    if degradation in (None, "clean"):
        return img
    if degradation == "jpeg_q30":
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=30)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if degradation == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    if degradation == "motion_blur":
        arr = np.asarray(img).astype(np.float32)
        kernel_size = 9
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        import cv2

        arr = cv2.filter2D(arr, -1, kernel)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if degradation == "sensor_noise":
        arr = np.asarray(img).astype(np.float32)
        noise = rng.normal(0, 8, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if degradation == "low_light":
        return ImageEnhance.Brightness(img).enhance(0.5)
    raise ValueError(f"Unknown degradation: {degradation}")


def safe_torch_load(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_model(model_path, model_name, device, frequency_branch=False, frequency_mode="auto", frequency_features=128):
    checkpoint = safe_torch_load(model_path, device)
    checkpoint_threshold = None
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        checkpoint_threshold = checkpoint.get("threshold")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        checkpoint_threshold = checkpoint.get("threshold")
    else:
        state = checkpoint

    if frequency_mode == "auto":
        frequency_mode = checkpoint_config.get("frequency_mode", "laplacian" if frequency_branch else "none")
    checkpoint_frequency_branch = bool(checkpoint_config.get("frequency_branch", False))
    frequency_branch = bool(frequency_branch or checkpoint_frequency_branch or frequency_mode != "none")
    if frequency_branch and frequency_mode == "none":
        frequency_mode = "laplacian"
    if not frequency_branch:
        frequency_mode = "none"
    frequency_features = int(checkpoint_config.get("frequency_features", frequency_features))

    model = DeepfakeEfficientNet(
        model_name=model_name,
        pretrained=False,
        frequency_branch=frequency_branch,
        frequency_features=frequency_features,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint_threshold, frequency_branch, frequency_mode


def tensor_to_numpy(outputs):
    return torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)


def move_inputs_to_device(batch_inputs, device):
    if isinstance(batch_inputs, (list, tuple)):
        images = batch_inputs[0].to(device)
        frequency = batch_inputs[1].to(device)
        return images, frequency
    return batch_inputs.to(device), None


def forward_model(model, images, frequency=None):
    if frequency is None:
        return model(images)
    return model(images, frequency=frequency)


def predict_dataset(model, split_file, transform, device, batch_size, num_workers, degradation=None, limit=None, frequency_mode="none"):
    dataset = EvalDataset(
        split_file,
        transform=transform,
        degradation=degradation,
        limit=limit,
        frequency_mode=frequency_mode,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    probs = []
    paths = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch_inputs, batch_labels, batch_paths in loader:
            images, frequency = move_inputs_to_device(batch_inputs, device)
            outputs = forward_model(model, images, frequency)
            probs.extend(tensor_to_numpy(outputs).tolist())
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())
            paths.extend(list(batch_paths))
    elapsed = time.perf_counter() - start
    df = pd.DataFrame(
        {
            "path": paths,
            "label": labels,
            "prob_fake": probs,
            "source": [source_name(p) for p in paths],
        }
    )
    return df, elapsed


def calc_eer(y_true, y_prob):
    if len(set(y_true)) < 2:
        return None, None
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def finite(value):
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
    except TypeError:
        pass
    return float(value)


def threshold_for_predictions(spec, checkpoint_threshold, fixed_threshold, y_true, y_prob):
    eer, eer_threshold = calc_eer(y_true, y_prob)
    if spec == "eer":
        threshold = eer_threshold if eer_threshold is not None else 0.5
    elif spec == "checkpoint":
        threshold = checkpoint_threshold if checkpoint_threshold is not None else fixed_threshold
    elif spec == "fixed":
        threshold = fixed_threshold
    else:
        raise ValueError(f"Unknown threshold mode: {spec}")
    return float(threshold), eer, eer_threshold


def calculate_metrics(df, dataset, condition, threshold, eer, eer_threshold, elapsed):
    y_true = df["label"].to_numpy(dtype=int)
    y_prob = df["prob_fake"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    df["pred"] = y_pred
    df["correct"] = y_pred == y_true
    df["error_type"] = np.where(
        (y_true == 0) & (y_pred == 1),
        "FP",
        np.where((y_true == 1) & (y_pred == 0), "FN", ""),
    )

    if len(set(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        auc_roc = roc_auc_score(y_true, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(recall_curve, precision_curve)
        avg_precision = average_precision_score(y_true, y_prob)
        specificity = tn / (tn + fp) if (tn + fp) else None
        far = fp / (fp + tn) if (fp + tn) else None
        frr = fn / (fn + tp) if (fn + tp) else None
        mcc = matthews_corrcoef(y_true, y_pred)
    else:
        tn = fp = fn = tp = None
        auc_roc = auc_pr = avg_precision = specificity = far = frr = mcc = None

    samples = len(df)
    row = MetricRow(
        dataset=dataset,
        condition=condition,
        samples=samples,
        real=int((y_true == 0).sum()),
        fake=int((y_true == 1).sum()),
        threshold=float(threshold),
        eer_threshold=finite(eer_threshold),
        accuracy=finite(accuracy_score(y_true, y_pred)) if samples else None,
        balanced_accuracy=finite(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) == 2 else None,
        precision=finite(precision_score(y_true, y_pred, zero_division=0)) if samples else None,
        recall=finite(recall_score(y_true, y_pred, zero_division=0)) if samples else None,
        specificity=finite(specificity),
        f1=finite(f1_score(y_true, y_pred, zero_division=0)) if samples else None,
        auc_roc=finite(auc_roc),
        auc_pr=finite(auc_pr),
        average_precision=finite(avg_precision),
        eer=finite(eer),
        far=finite(far),
        frr=finite(frr),
        mcc=finite(mcc),
        tn=None if tn is None else int(tn),
        fp=None if fp is None else int(fp),
        fn=None if fn is None else int(fn),
        tp=None if tp is None else int(tp),
        elapsed_sec=float(elapsed),
        images_per_sec=finite(samples / elapsed) if elapsed > 0 and samples else None,
        ms_per_image=finite((elapsed / samples) * 1000.0) if samples else None,
    )
    return row, df


def save_confusion_matrix(df, out_path, title):
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = df["pred"].to_numpy(dtype=int)
    if len(set(y_true)) < 2:
        return
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    labels = np.array(
        [
            f"{cm[0, 0]}\n({cm_pct[0, 0]:.2%})",
            f"{cm[0, 1]}\n({cm_pct[0, 1]:.2%})",
            f"{cm[1, 0]}\n({cm_pct[1, 0]:.2%})",
            f"{cm[1, 1]}\n({cm_pct[1, 1]:.2%})",
        ]
    ).reshape(2, 2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Pred Real", "Pred Fake"],
        yticklabels=["Actual Real", "Actual Fake"],
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_roc_curve(df, out_path, title):
    y_true = df["label"].to_numpy(dtype=int)
    y_prob = df["prob_fake"].to_numpy(dtype=float)
    if len(set(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = roc_auc_score(y_true, y_prob)
    eer, _ = calc_eer(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    if eer is not None:
        plt.scatter([eer], [1 - eer], color="red", label=f"EER = {eer:.4f}")
    plt.xlabel("False Positive Rate / FAR")
    plt.ylabel("True Positive Rate / 1-FRR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_pr_curve(df, out_path, title):
    y_true = df["label"].to_numpy(dtype=int)
    y_prob = df["prob_fake"].to_numpy(dtype=float)
    if len(set(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    score = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Average precision = {score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_error_samples(df, out_dir, max_per_type):
    if max_per_type <= 0:
        return
    out_dir = Path(out_dir)
    fp_dir = out_dir / "false_positives"
    fn_dir = out_dir / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    fp = df[df["error_type"] == "FP"].copy()
    fn = df[df["error_type"] == "FN"].copy()
    fp = fp.sort_values("prob_fake", ascending=False).head(max_per_type)
    fn = fn.sort_values("prob_fake", ascending=True).head(max_per_type)

    for _, row in fp.iterrows():
        copy_sample(row, fp_dir)
    for _, row in fn.iterrows():
        copy_sample(row, fn_dir)


def copy_sample(row, dst_dir):
    src = Path(row["path"])
    if not src.exists():
        return
    prob = float(row["prob_fake"])
    prefix = f"{row.name:06d}_p{prob:.4f}_label{int(row['label'])}_pred{int(row['pred'])}"
    dst = dst_dir / f"{prefix}_{src.name}"
    try:
        shutil.copy2(src, dst)
    except OSError:
        pass


def split_summary(split_file, name):
    if not split_file or not Path(split_file).exists():
        return []
    samples = read_split(split_file)
    rows = []
    by_source = defaultdict(list)
    for path, label in samples:
        by_source[source_name(path)].append((path, label))
    for source, items in sorted(by_source.items()):
        labels = Counter(label for _, label in items)
        rows.append(
            {
                "split": name,
                "source": source,
                "samples": len(items),
                "real": labels.get(0, 0),
                "fake": labels.get(1, 0),
            }
        )
    labels = Counter(label for _, label in samples)
    rows.append(
        {
            "split": name,
            "source": "ALL",
            "samples": len(samples),
            "real": labels.get(0, 0),
            "fake": labels.get(1, 0),
        }
    )
    return rows


def evaluate_and_save(
    model,
    split_file,
    dataset_name,
    condition,
    args,
    transform,
    checkpoint_threshold,
    out_dir,
    degradation=None,
    write_predictions=True,
    make_plots=True,
):
    pred_df, elapsed = predict_dataset(
        model=model,
        split_file=split_file,
        transform=transform,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        degradation=degradation,
        limit=args.max_samples,
        frequency_mode=args.frequency_mode,
    )
    threshold, eer, eer_threshold = threshold_for_predictions(
        args.threshold_mode,
        checkpoint_threshold,
        args.fixed_threshold,
        pred_df["label"].to_numpy(dtype=int),
        pred_df["prob_fake"].to_numpy(dtype=float),
    )
    row, pred_df = calculate_metrics(pred_df, dataset_name, condition, threshold, eer, eer_threshold, elapsed)

    stem = safe_stem(f"{dataset_name}_{condition}")
    if write_predictions:
        pred_df.to_csv(out_dir / f"{stem}_predictions.csv", index=False)
        report = classification_report(
            pred_df["label"].to_numpy(dtype=int),
            pred_df["pred"].to_numpy(dtype=int),
            labels=[0, 1],
            target_names=["Real", "Fake"],
            zero_division=0,
        )
        (out_dir / f"{stem}_classification_report.txt").write_text(report, encoding="utf-8")
    if make_plots:
        save_confusion_matrix(pred_df, out_dir / f"{stem}_confusion_matrix.png", f"{dataset_name} - {condition}")
        save_roc_curve(pred_df, out_dir / f"{stem}_roc_curve.png", f"{dataset_name} - {condition}")
        save_pr_curve(pred_df, out_dir / f"{stem}_pr_curve.png", f"{dataset_name} - {condition}")
    return row, pred_df


def safe_stem(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def write_markdown_report(out_dir, args, metric_rows, split_rows, leakage, model_path, checkpoint_threshold):
    lines = []
    lines.append("# Deepfake Model Evaluation Report")
    lines.append("")
    lines.append(f"- Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Model name: `{args.model_name}`")
    lines.append(f"- Test split: `{args.test}`")
    lines.append(f"- Threshold mode: `{args.threshold_mode}`")
    lines.append(f"- Fixed fallback threshold: `{args.fixed_threshold}`")
    lines.append(f"- Checkpoint threshold: `{checkpoint_threshold}`")
    lines.append(f"- Device: `{args.device}`")
    lines.append(f"- Frequency branch: `{getattr(args, 'frequency_branch', False)}`")
    lines.append(f"- Frequency mode: `{getattr(args, 'frequency_mode', 'none')}`")
    lines.append("")

    if split_rows:
        lines.append("## Split Summary")
        lines.append("")
        lines.append("| Split | Source | Samples | Real | Fake |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in split_rows:
            lines.append(f"| {row['split']} | {row['source']} | {row['samples']} | {row['real']} | {row['fake']} |")
        lines.append("")

    if leakage:
        lines.append("## Identity Leakage Audit")
        lines.append("")
        lines.append(f"- Train unique IDs: {leakage.get('train_unique_ids', 0)}")
        for item in leakage.get("comparisons", []):
            lines.append(
                f"- {item['split']}: unique IDs={item['unique_ids']}, leaked IDs={item['leaked_ids']}, "
                f"leakage={item['leakage_percent']:.2f}%"
            )
        lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| Dataset | Condition | Samples | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | EER | FAR | FRR | TP | FP | TN | FN | ms/img |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in metric_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    row.condition,
                    str(row.samples),
                    fmt(row.accuracy),
                    fmt(row.precision),
                    fmt(row.recall),
                    fmt(row.f1),
                    fmt(row.auc_roc),
                    fmt(row.auc_pr),
                    fmt(row.eer),
                    fmt(row.far),
                    fmt(row.frr),
                    str(row.tp),
                    str(row.fp),
                    str(row.tn),
                    str(row.fn),
                    fmt(row.ms_per_image),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `summary_metrics.csv`: compact metrics table")
    lines.append("- `summary.json`: full structured metadata")
    lines.append("- `*_predictions.csv`: raw probability, label, prediction, source, error type")
    lines.append("- `*_confusion_matrix.png`, `*_roc_curve.png`, `*_pr_curve.png`: figures for report")
    lines.append("- `error_analysis/`: copied high-confidence FP/FN samples, if enabled")
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def fmt(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"


def parse_external_split(value):
    if ":" in value and not (len(value) > 1 and value[1] == ":"):
        name, path = value.split(":", 1)
        return name, path
    path = value
    return Path(path).stem, path


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for the deepfake detector.")
    parser.add_argument("--model", default="models/best_pytorch_model_final.pth")
    parser.add_argument("--model-name", default="efficientnet_b4")
    parser.add_argument("--test", default="splits/test.txt")
    parser.add_argument("--train", default="splits/train.txt")
    parser.add_argument("--val", default="splits/val.txt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit per split.")
    parser.add_argument("--threshold-mode", choices=["checkpoint", "eer", "fixed"], default="checkpoint")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--robustness", action="store_true", help="Run JPEG/blur/noise/low-light stress tests.")
    parser.add_argument("--skip-dfdc", action="store_true", help="Do not auto-evaluate splits/dfdc_test.txt.")
    parser.add_argument(
        "--external-split",
        action="append",
        default=[],
        help="Extra split as NAME:path or path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--external-dir",
        action="append",
        default=[],
        help="Directory with real/fake subfolders. Can be passed multiple times.",
    )
    parser.add_argument("--copy-errors", type=int, default=40, help="Copy top FP/FN samples per error type.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--frequency-branch", action="store_true", help="Force frequency-branch model evaluation.")
    parser.add_argument(
        "--frequency-mode",
        choices=["auto", "none", "laplacian", "fft"],
        default="auto",
        help="Frequency preprocessing mode. auto reads checkpoint config.",
    )
    parser.add_argument("--frequency-features", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    out_dir = Path(args.out) if args.out else Path("evaluation_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms(is_train=False)
    model, checkpoint_threshold, frequency_branch, frequency_mode = load_model(
        args.model,
        args.model_name,
        args.device,
        frequency_branch=args.frequency_branch,
        frequency_mode=args.frequency_mode,
        frequency_features=args.frequency_features,
    )
    args.frequency_branch = frequency_branch
    args.frequency_mode = frequency_mode
    print(f"Model frequency branch: {args.frequency_branch} ({args.frequency_mode})")

    split_rows = []
    split_rows.extend(split_summary(args.train, "train"))
    split_rows.extend(split_summary(args.val, "val"))
    split_rows.extend(split_summary(args.test, "test"))
    pd.DataFrame(split_rows).to_csv(out_dir / "split_summary.csv", index=False)

    leakage = identity_leakage_report(args.train, args.val, args.test)
    (out_dir / "identity_leakage.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_rows = []

    print(f"Evaluating internal test split: {args.test}")
    internal_row, internal_pred = evaluate_and_save(
        model,
        args.test,
        "internal_test",
        "clean",
        args,
        transform,
        checkpoint_threshold,
        out_dir,
        make_plots=not args.no_plots,
    )
    metric_rows.append(internal_row)
    save_error_samples(internal_pred, out_dir / "error_analysis", args.copy_errors)

    source_rows = []
    for source, group in internal_pred.groupby("source"):
        if group["label"].nunique() < 2:
            continue
        threshold, eer, eer_threshold = threshold_for_predictions(
            args.threshold_mode,
            checkpoint_threshold,
            args.fixed_threshold,
            group["label"].to_numpy(dtype=int),
            group["prob_fake"].to_numpy(dtype=float),
        )
        row, group = calculate_metrics(group.copy(), "internal_by_source", source, threshold, eer, eer_threshold, 0.0)
        source_rows.append(row)
        metric_rows.append(row)
    pd.DataFrame([asdict(r) for r in source_rows]).to_csv(out_dir / "cross_source_metrics.csv", index=False)

    if args.robustness:
        print("Running robustness stress tests...")
        robust_rows = []
        for condition in ["jpeg_q30", "gaussian_blur", "motion_blur", "sensor_noise", "low_light"]:
            row, _ = evaluate_and_save(
                model,
                args.test,
                "internal_test",
                condition,
                args,
                transform,
                checkpoint_threshold,
                out_dir,
                degradation=condition,
                write_predictions=False,
                make_plots=False,
            )
            robust_rows.append(row)
            metric_rows.append(row)
        pd.DataFrame([asdict(r) for r in robust_rows]).to_csv(out_dir / "robustness_metrics.csv", index=False)

    external_items = list(args.external_split)
    dfdc_path = Path("splits/dfdc_test.txt")
    if dfdc_path.exists() and not args.skip_dfdc:
        external_items.append(f"DFDC:{dfdc_path}")

    external_rows = []
    for value in external_items:
        name, split_path = parse_external_split(value)
        if not Path(split_path).exists():
            print(f"Skipping missing external split: {split_path}")
            continue
        print(f"Evaluating external split: {name} -> {split_path}")
        row, _ = evaluate_and_save(
            model,
            split_path,
            f"external_{name}",
            "clean",
            args,
            transform,
            checkpoint_threshold,
            out_dir,
            make_plots=not args.no_plots,
        )
        external_rows.append(row)
        metric_rows.append(row)

    for directory in args.external_dir:
        directory = Path(directory)
        name = directory.name
        split_path = out_dir / f"external_dir_{safe_stem(name)}.txt"
        samples = create_split_from_dir(directory, split_path)
        if not samples:
            print(f"Skipping external dir with no labeled images: {directory}")
            continue
        print(f"Evaluating external dir: {directory} ({len(samples)} images)")
        row, _ = evaluate_and_save(
            model,
            split_path,
            f"external_{name}",
            "clean",
            args,
            transform,
            checkpoint_threshold,
            out_dir,
            make_plots=not args.no_plots,
        )
        external_rows.append(row)
        metric_rows.append(row)

    if external_rows:
        pd.DataFrame([asdict(r) for r in external_rows]).to_csv(out_dir / "external_metrics.csv", index=False)

    metrics_df = pd.DataFrame([asdict(r) for r in metric_rows])
    metrics_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(args.model),
        "model_name": args.model_name,
        "checkpoint_threshold": checkpoint_threshold,
        "threshold_mode": args.threshold_mode,
        "fixed_threshold": args.fixed_threshold,
        "frequency_branch": bool(args.frequency_branch),
        "frequency_mode": args.frequency_mode,
        "device": str(args.device),
        "args": {**vars(args), "device": str(args.device)},
        "metrics": [asdict(r) for r in metric_rows],
        "identity_leakage": leakage,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(out_dir, args, metric_rows, split_rows, leakage, args.model, checkpoint_threshold)

    print("")
    print("=" * 80)
    print(f"Evaluation complete: {out_dir}")
    print("Main files:")
    print(f"- {out_dir / 'REPORT.md'}")
    print(f"- {out_dir / 'summary_metrics.csv'}")
    print(f"- {out_dir / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
