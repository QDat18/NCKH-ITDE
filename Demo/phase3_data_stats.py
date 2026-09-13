import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LABEL_NAMES = {0: "Real", 1: "Fake"}


def parse_split_line(line):
    text = line.strip()
    if not text:
        return None
    if "," in text:
        path, label = text.rsplit(",", 1)
    else:
        path, label = text.rsplit(None, 1)
    return path.strip(), int(label.strip())


def read_split(split_file, split_name, base_dir):
    rows = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parsed = parse_split_line(line)
            if parsed is None:
                continue
            raw_path, label = parsed
            source = source_name(raw_path)
            group = video_group_key(raw_path)
            resolved, exists = resolve_existing_path(raw_path, base_dir)
            rows.append(
                {
                    "split": split_name,
                    "line_no": line_no,
                    "path": raw_path,
                    "resolved_path": str(resolved),
                    "exists": exists,
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, str(label)),
                    "source": source,
                    "group": group,
                    "identity_ids": sorted(identity_ids(raw_path)),
                }
            )
    return rows


def resolve_existing_path(raw_path, base_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path, path.exists()
    resolved = base_dir / raw_path
    return resolved, resolved.exists()


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


def video_group_key(path):
    base = os.path.basename(str(path))
    stem = re.sub(r"\.[^.]+$", "", base)
    stem = re.sub(r"_f\d+$", "", stem)
    return f"{source_name(path)}::{stem}"


def identity_ids(path):
    ids = set()
    base = os.path.basename(str(path))
    if "FF_" in base:
        for part in base.split("_"):
            if part.isdigit() and len(part) >= 2:
                ids.add(f"FF_{part}")
    elif "Celeb-" in base:
        for part in base.split("_"):
            if part.startswith("id") and part[2:].isdigit():
                ids.add(f"Celeb_{part}")
    return ids


def pct(value, total):
    return 0.0 if total == 0 else 100.0 * value / total


def safe_ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def split_source_label_summary(samples_by_split):
    rows = []
    for split_name, samples in samples_by_split.items():
        total = len(samples)
        by_source = defaultdict(list)
        for sample in samples:
            by_source[sample["source"]].append(sample)

        for source, items in sorted(by_source.items()):
            rows.append(summary_row(split_name, source, items, total))
        rows.append(summary_row(split_name, "ALL", samples, total))
    return rows


def summary_row(split_name, source, samples, split_total):
    labels = Counter(sample["label"] for sample in samples)
    groups = defaultdict(int)
    identities = set()
    existing = 0
    for sample in samples:
        groups[sample["group"]] += 1
        identities.update(sample["identity_ids"])
        if sample["exists"]:
            existing += 1
    group_sizes = list(groups.values())
    real = labels.get(0, 0)
    fake = labels.get(1, 0)
    return {
        "split": split_name,
        "source": source,
        "samples": len(samples),
        "source_pct_in_split": pct(len(samples), split_total),
        "real": real,
        "real_pct_in_row": pct(real, len(samples)),
        "fake": fake,
        "fake_pct_in_row": pct(fake, len(samples)),
        "fake_to_real_ratio": safe_ratio(fake, real),
        "groups": len(groups),
        "avg_samples_per_group": safe_ratio(len(samples), len(groups)),
        "median_samples_per_group": median(group_sizes),
        "max_samples_per_group": max(group_sizes) if group_sizes else 0,
        "identity_ids": len(identities),
        "existing": existing,
        "missing": len(samples) - existing,
    }


def label_balance_summary(samples_by_split):
    rows = []
    for split_name, samples in samples_by_split.items():
        total = len(samples)
        labels = Counter(sample["label"] for sample in samples)
        for label in sorted(labels):
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, str(label)),
                    "samples": labels[label],
                    "pct_in_split": pct(labels[label], total),
                }
            )
    return rows


def source_balance_summary(samples_by_split):
    rows = []
    for split_name, samples in samples_by_split.items():
        total = len(samples)
        sources = Counter(sample["source"] for sample in samples)
        for source, count in sorted(sources.items()):
            rows.append(
                {
                    "split": split_name,
                    "source": source,
                    "samples": count,
                    "pct_in_split": pct(count, total),
                }
            )
    return rows


def group_summary(samples_by_split):
    rows = []
    for split_name, samples in samples_by_split.items():
        by_source = defaultdict(lambda: defaultdict(int))
        for sample in samples:
            by_source[sample["source"]][sample["group"]] += 1
        for source, groups in sorted(by_source.items()):
            sizes = list(groups.values())
            rows.append(
                {
                    "split": split_name,
                    "source": source,
                    "groups": len(sizes),
                    "samples": sum(sizes),
                    "min_samples_per_group": min(sizes) if sizes else 0,
                    "median_samples_per_group": median(sizes),
                    "avg_samples_per_group": safe_ratio(sum(sizes), len(sizes)),
                    "max_samples_per_group": max(sizes) if sizes else 0,
                }
            )
    return rows


def overlap_audit(samples_by_split):
    rows = []
    names = list(samples_by_split)
    group_sets = {
        split: {sample["group"] for sample in samples}
        for split, samples in samples_by_split.items()
    }
    id_sets = {}
    for split, samples in samples_by_split.items():
        ids = set()
        for sample in samples:
            ids.update(sample["identity_ids"])
        id_sets[split] = ids

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            group_overlap = group_sets[left].intersection(group_sets[right])
            id_overlap = id_sets[left].intersection(id_sets[right])
            rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "left_groups": len(group_sets[left]),
                    "right_groups": len(group_sets[right]),
                    "overlap_groups": len(group_overlap),
                    "overlap_group_examples": ";".join(sorted(group_overlap)[:20]),
                    "left_identity_ids": len(id_sets[left]),
                    "right_identity_ids": len(id_sets[right]),
                    "overlap_identity_ids": len(id_overlap),
                    "overlap_identity_examples": ";".join(sorted(id_overlap)[:20]),
                }
            )
    return rows


def train_sampling_plan(train_samples):
    total = len(train_samples)
    label_counts = Counter(sample["label"] for sample in train_samples)
    source_counts = Counter(sample["source"] for sample in train_samples)
    source_label_counts = Counter((sample["source"], sample["label"]) for sample in train_samples)

    num_labels = len(label_counts) or 1
    num_sources = len(source_counts) or 1
    num_source_label_cells = len(source_label_counts) or 1

    rows = []
    for (source, label), count in sorted(source_label_counts.items()):
        label_count = label_counts[label]
        source_count = source_counts[source]
        label_weight = total / (num_labels * label_count) if label_count else 0.0
        source_weight = total / (num_sources * source_count) if source_count else 0.0
        source_label_weight = total / (num_source_label_cells * count) if count else 0.0
        combined_sqrt_weight = math.sqrt(label_weight * source_weight)
        rows.append(
            {
                "source": source,
                "label": label,
                "label_name": LABEL_NAMES.get(label, str(label)),
                "samples": count,
                "pct_of_train": pct(count, total),
                "source_total": source_count,
                "label_total": label_count,
                "label_weight": label_weight,
                "source_weight": source_weight,
                "source_label_weight": source_label_weight,
                "combined_sqrt_weight": combined_sqrt_weight,
            }
        )
    return rows


def train_sampler_json(train_samples):
    plan = train_sampling_plan(train_samples)
    return {
        "strategy_notes": [
            "Use label_weight for class-balanced sampling.",
            "Use source_label_weight for stronger domain-and-label balancing.",
            "Use combined_sqrt_weight as a gentler compromise if source_label_weight over-samples very small cells.",
        ],
        "source_label_weights": {
            f"{row['source']}|{row['label']}": row["source_label_weight"]
            for row in plan
        },
        "combined_sqrt_weights": {
            f"{row['source']}|{row['label']}": row["combined_sqrt_weight"]
            for row in plan
        },
        "label_weights": {
            str(label): len(train_samples) / (len(Counter(sample["label"] for sample in train_samples)) * count)
            for label, count in Counter(sample["label"] for sample in train_samples).items()
        },
    }


def recommendations(samples_by_split, overlaps, train_plan):
    lines = []
    train = samples_by_split.get("train", [])
    labels = Counter(sample["label"] for sample in train)
    sources = Counter(sample["source"] for sample in train)
    real = labels.get(0, 0)
    fake = labels.get(1, 0)
    ratio = safe_ratio(fake, real)
    if ratio and ratio > 2:
        lines.append(
            f"Train split is class-imbalanced: fake/real ratio is {ratio:.2f}. Use Focal Loss plus label-balanced or source-label-balanced sampling."
        )
    if sources:
        max_source, max_count = sources.most_common(1)[0]
        max_pct = pct(max_count, len(train))
        if max_pct > 60:
            lines.append(
                f"Train split is source-imbalanced: {max_source} accounts for {max_pct:.1f}% of train samples. Prefer domain-balanced sampling."
            )
    risky = [row for row in overlaps if row["overlap_groups"] or row["overlap_identity_ids"]]
    if risky:
        lines.append("Overlap was detected. Do not train until split leakage is fixed.")
    else:
        lines.append("No video-group or identity overlap detected among configured splits.")
    tiny_cells = [row for row in train_plan if row["samples"] < 100]
    if tiny_cells:
        names = ", ".join(f"{row['source']} {row['label_name']}={row['samples']}" for row in tiny_cells)
        lines.append(f"Some source-label cells are small: {names}. Avoid overly aggressive oversampling without augmentation.")
    return lines


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, args, summary_rows, label_rows, source_rows, overlap_rows, sampling_rows, recommendation_lines):
    lines = []
    lines.append("# Phase 3 Data Statistics Report")
    lines.append("")
    lines.append(f"- Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Train split: `{args.train}`")
    lines.append(f"- Val split: `{args.val}`")
    lines.append(f"- Test split: `{args.test}`")
    lines.append(f"- External split: `{args.external}`")
    lines.append("")

    lines.append("## Split Summary")
    lines.append("")
    lines.append("| Split | Source | Samples | Real | Fake | Fake/Real | Source % | Groups | Missing |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['source']} | {row['samples']} | {row['real']} | {row['fake']} | "
            f"{fmt(row['fake_to_real_ratio'], 2)} | {fmt(row['source_pct_in_split'], 1)} | {row['groups']} | {row['missing']} |"
        )
    lines.append("")

    lines.append("## Label Balance")
    lines.append("")
    lines.append("| Split | Label | Samples | Percent |")
    lines.append("|---|---|---:|---:|")
    for row in label_rows:
        lines.append(f"| {row['split']} | {row['label_name']} | {row['samples']} | {fmt(row['pct_in_split'], 1)}% |")
    lines.append("")

    lines.append("## Source Balance")
    lines.append("")
    lines.append("| Split | Source | Samples | Percent |")
    lines.append("|---|---|---:|---:|")
    for row in source_rows:
        lines.append(f"| {row['split']} | {row['source']} | {row['samples']} | {fmt(row['pct_in_split'], 1)}% |")
    lines.append("")

    lines.append("## Overlap Audit")
    lines.append("")
    lines.append("| Left | Right | Overlap Groups | Overlap Identity IDs |")
    lines.append("|---|---|---:|---:|")
    for row in overlap_rows:
        lines.append(
            f"| {row['left_split']} | {row['right_split']} | {row['overlap_groups']} | {row['overlap_identity_ids']} |"
        )
    lines.append("")

    lines.append("## Train Sampling Plan")
    lines.append("")
    lines.append("| Source | Label | Samples | Train % | Label Weight | Source Weight | Source-Label Weight | Gentle Combined Weight |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in sampling_rows:
        lines.append(
            f"| {row['source']} | {row['label_name']} | {row['samples']} | {fmt(row['pct_of_train'], 1)}% | "
            f"{fmt(row['label_weight'], 4)} | {fmt(row['source_weight'], 4)} | "
            f"{fmt(row['source_label_weight'], 4)} | {fmt(row['combined_sqrt_weight'], 4)} |"
        )
    lines.append("")

    lines.append("## Recommendations For Phase 3")
    lines.append("")
    for item in recommendation_lines:
        lines.append(f"- {item}")
    lines.append("- Train with `train_v2.txt` and `val_v2.txt`, then evaluate on `test_v2.txt` and `dfdc_holdout_test.txt`.")
    lines.append("- Keep `dfdc_holdout_test.txt` out of training, threshold selection, and early stopping.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 3 data statistics and sampler planning files.")
    parser.add_argument("--base-dir", default=str(SCRIPT_DIR), help="Base directory for resolving relative split paths.")
    parser.add_argument("--train", default="splits/v2/train_v2.txt")
    parser.add_argument("--val", default="splits/v2/val_v2.txt")
    parser.add_argument("--test", default="splits/v2/test_v2.txt")
    parser.add_argument("--external", default="splits/v2/dfdc_holdout_test.txt")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "phase3_outputs" / ("data_stats_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": base_dir / args.train,
        "val": base_dir / args.val,
        "test": base_dir / args.test,
        "external": base_dir / args.external,
    }
    samples_by_split = {}
    for split_name, split_path in split_files.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Split not found: {split_path}")
        samples_by_split[split_name] = read_split(split_path, split_name, base_dir)

    summary_rows = split_source_label_summary(samples_by_split)
    label_rows = label_balance_summary(samples_by_split)
    source_rows = source_balance_summary(samples_by_split)
    group_rows = group_summary(samples_by_split)
    overlap_rows = overlap_audit(samples_by_split)
    sampling_rows = train_sampling_plan(samples_by_split["train"])
    sampler_config = train_sampler_json(samples_by_split["train"])
    recommendation_lines = recommendations(samples_by_split, overlap_rows, sampling_rows)

    write_csv(
        out_dir / "split_source_label_summary.csv",
        summary_rows,
        [
            "split",
            "source",
            "samples",
            "source_pct_in_split",
            "real",
            "real_pct_in_row",
            "fake",
            "fake_pct_in_row",
            "fake_to_real_ratio",
            "groups",
            "avg_samples_per_group",
            "median_samples_per_group",
            "max_samples_per_group",
            "identity_ids",
            "existing",
            "missing",
        ],
    )
    write_csv(out_dir / "label_balance_summary.csv", label_rows, ["split", "label", "label_name", "samples", "pct_in_split"])
    write_csv(out_dir / "source_balance_summary.csv", source_rows, ["split", "source", "samples", "pct_in_split"])
    write_csv(
        out_dir / "group_summary.csv",
        group_rows,
        [
            "split",
            "source",
            "groups",
            "samples",
            "min_samples_per_group",
            "median_samples_per_group",
            "avg_samples_per_group",
            "max_samples_per_group",
        ],
    )
    write_csv(
        out_dir / "split_overlap_audit.csv",
        overlap_rows,
        [
            "left_split",
            "right_split",
            "left_groups",
            "right_groups",
            "overlap_groups",
            "overlap_group_examples",
            "left_identity_ids",
            "right_identity_ids",
            "overlap_identity_ids",
            "overlap_identity_examples",
        ],
    )
    write_csv(
        out_dir / "train_sampling_plan.csv",
        sampling_rows,
        [
            "source",
            "label",
            "label_name",
            "samples",
            "pct_of_train",
            "source_total",
            "label_total",
            "label_weight",
            "source_weight",
            "source_label_weight",
            "combined_sqrt_weight",
        ],
    )
    (out_dir / "sampler_config.json").write_text(json.dumps(sampler_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "phase3_data_stats.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "splits": {name: str(path) for name, path in split_files.items()},
                "summary": summary_rows,
                "label_balance": label_rows,
                "source_balance": source_rows,
                "group_summary": group_rows,
                "overlap": overlap_rows,
                "train_sampling_plan": sampling_rows,
                "recommendations": recommendation_lines,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(out_dir / "PHASE3_DATA_STATS.md", args, summary_rows, label_rows, source_rows, overlap_rows, sampling_rows, recommendation_lines)

    print(f"Phase 3 data statistics complete: {out_dir}")
    print(f"- {out_dir / 'PHASE3_DATA_STATS.md'}")
    print(f"- {out_dir / 'split_source_label_summary.csv'}")
    print(f"- {out_dir / 'train_sampling_plan.csv'}")
    print(f"- {out_dir / 'sampler_config.json'}")


if __name__ == "__main__":
    main()
