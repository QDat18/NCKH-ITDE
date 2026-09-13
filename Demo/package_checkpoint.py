import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch


def safe_torch_load(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def pick_best_row(log_csv, epoch=None, metric=None):
    with open(log_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {log_csv}")

    if epoch is not None:
        for row in rows:
            if int(float(row["epoch"])) == epoch:
                return row
        raise ValueError(f"Epoch {epoch} not found in {log_csv}")

    if metric is None:
        sample = rows[0]
        metric = "selection_score" if "selection_score" in sample else "val_auc"
    return max(rows, key=lambda row: float(row[metric]))


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def main():
    parser = argparse.ArgumentParser(description="Wrap a raw PyTorch state_dict with V2 evaluation metadata.")
    parser.add_argument("--model", default="models/best_pytorch_model_b4_v2.pth")
    parser.add_argument("--log", default="training_log_v2.csv")
    parser.add_argument("--out", default="models/best_pytorch_model_b4_v2_packaged.pth")
    parser.add_argument("--model-name", default="efficientnet_b4")
    parser.add_argument("--train", default="splits/v2/train_v2.txt")
    parser.add_argument("--val", default="splits/v2/val_v2.txt")
    parser.add_argument("--test", default="splits/v2/test_v2.txt")
    parser.add_argument("--sampler", default="source_label")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Package a specific epoch from the log. Default: best selection_score if present, else val_auc.",
    )
    parser.add_argument("--metric", default=None, help="Metric used to pick the best row. Default: selection_score if present, else val_auc.")
    args = parser.parse_args()

    best = pick_best_row(args.log, args.epoch, args.metric)
    checkpoint = safe_torch_load(args.model)
    state_dict = extract_state_dict(checkpoint)

    payload = {
        "state_dict": state_dict,
        "model_name": args.model_name,
        "threshold": float(best["threshold"]),
        "epoch": int(float(best["epoch"])),
        "metrics": {
            "val_auc": float_or_none(best.get("val_auc")),
            "val_eer": float_or_none(best.get("val_eer")),
            "val_acc": float_or_none(best.get("val_acc")),
            "val_precision": float_or_none(best.get("val_precision")),
            "val_recall": float_or_none(best.get("val_recall")),
            "val_f1": float_or_none(best.get("val_f1")),
            "val_f2": float_or_none(best.get("val_f2")),
            "selection_score": float_or_none(best.get("selection_score")),
            "train_loss": float_or_none(best.get("train_loss")),
            "val_loss": float_or_none(best.get("val_loss")),
        },
        "splits": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
        },
        "config": {
            "sampler": args.sampler,
        },
        "source_checkpoint": args.model,
        "source_log": args.log,
        "packaged_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"Packaged checkpoint: {out_path}")
    print(f"Best epoch: {payload['epoch']}")
    print(f"Validation AUC: {payload['metrics']['val_auc']:.6f}")
    if payload["metrics"].get("selection_score") is not None:
        print(f"Selection score: {payload['metrics']['selection_score']:.6f}")
    print(f"Threshold: {payload['threshold']:.6f}")


if __name__ == "__main__":
    main()
