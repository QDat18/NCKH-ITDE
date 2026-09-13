import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from dataset_pytorch import get_transforms, source_name
from evaluate_comprehensive import load_model, predict_dataset


def resolve_demo_path(path):
    if not path:
        return ""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return Path(__file__).resolve().parent / path_obj


def mine_hard_samples(pred_df, fake_max_prob, real_min_prob, hard_fake_weight, hard_real_weight, top_k_per_type):
    hard_fake = pred_df[(pred_df["label"] == 1) & (pred_df["prob_fake"] <= fake_max_prob)].copy()
    hard_fake["hard_type"] = "hard_fake_false_negative_risk"
    hard_fake["hard_weight"] = float(hard_fake_weight)
    hard_fake = hard_fake.sort_values("prob_fake", ascending=True)

    hard_real = pred_df[(pred_df["label"] == 0) & (pred_df["prob_fake"] >= real_min_prob)].copy()
    hard_real["hard_type"] = "hard_real_false_positive_risk"
    hard_real["hard_weight"] = float(hard_real_weight)
    hard_real = hard_real.sort_values("prob_fake", ascending=False)

    if top_k_per_type and top_k_per_type > 0:
        hard_fake = hard_fake.head(top_k_per_type)
        hard_real = hard_real.head(top_k_per_type)

    hard = pd.concat([hard_fake, hard_real], ignore_index=True)
    if hard.empty:
        return hard

    hard["source"] = hard["path"].map(source_name)
    return hard[
        [
            "path",
            "label",
            "prob_fake",
            "hard_type",
            "hard_weight",
            "source",
        ]
    ].sort_values(["hard_type", "source", "prob_fake"], ascending=[True, True, True])


def write_report(out_path, args, hard_df, pred_df):
    lines = []
    lines.append("# Hard Sample Mining Report")
    lines.append("")
    lines.append(f"- Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Fake FN-risk rule: `label=1 and prob_fake <= {args.fake_max_prob}`")
    lines.append(f"- Real FP-risk rule: `label=0 and prob_fake >= {args.real_min_prob}`")
    lines.append(f"- Total predictions: `{len(pred_df)}`")
    lines.append(f"- Hard samples: `{len(hard_df)}`")
    lines.append("")

    if hard_df.empty:
        lines.append("No hard samples matched the configured rules.")
    else:
        summary = (
            hard_df.groupby(["hard_type", "source", "label"])
            .size()
            .reset_index(name="samples")
            .sort_values(["hard_type", "source", "label"])
        )
        lines.append("## Summary")
        lines.append("")
        lines.append("| Hard type | Source | Label | Samples |")
        lines.append("|---|---|---:|---:|")
        for _, row in summary.iterrows():
            lines.append(f"| {row['hard_type']} | {row['source']} | {int(row['label'])} | {int(row['samples'])} |")

        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("- Use this CSV only with training/pool data, not held-out test data.")
        lines.append("- Hard Fake samples are oversampled to reduce false negatives.")
        lines.append("- Hard Real samples are also oversampled to protect precision while recall is improved.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard training samples for V3 recall improvement.")
    parser.add_argument("--model", default="models/best_pytorch_model_b4_v2_packaged.pth")
    parser.add_argument("--model-name", default="efficientnet_b4")
    parser.add_argument("--split", default="splits/v3/train_v3.txt")
    parser.add_argument("--out", default="splits/v3/hard_samples_v3.csv")
    parser.add_argument("--predictions-out", default="splits/v3/train_v3_predictions_for_mining.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fake-max-prob", type=float, default=0.50)
    parser.add_argument("--real-min-prob", type=float, default=0.50)
    parser.add_argument("--hard-fake-weight", type=float, default=2.5)
    parser.add_argument("--hard-real-weight", type=float, default=1.7)
    parser.add_argument("--top-k-per-type", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    model_path = resolve_demo_path(args.model)
    split_path = resolve_demo_path(args.split)
    out_path = resolve_demo_path(args.out)
    pred_out = resolve_demo_path(args.predictions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    transform = get_transforms(is_train=False)
    model, _ = load_model(model_path, args.model_name, device)
    pred_df, elapsed = predict_dataset(
        model=model,
        split_file=split_path,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pred_df.to_csv(pred_out, index=False)

    hard_df = mine_hard_samples(
        pred_df=pred_df,
        fake_max_prob=args.fake_max_prob,
        real_min_prob=args.real_min_prob,
        hard_fake_weight=args.hard_fake_weight,
        hard_real_weight=args.hard_real_weight,
        top_k_per_type=args.top_k_per_type,
    )
    hard_df.to_csv(out_path, index=False)
    report_path = out_path.with_suffix(".md")
    write_report(report_path, args, hard_df, pred_df)

    print(f"Predicted {len(pred_df)} samples in {elapsed:.2f}s on {device}.")
    print(f"Saved hard sample CSV: {out_path}")
    print(f"Saved hard sample report: {report_path}")


if __name__ == "__main__":
    main()
