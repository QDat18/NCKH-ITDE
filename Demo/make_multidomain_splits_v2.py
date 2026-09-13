import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_split_line(line):
    text = line.strip()
    if not text:
        return None
    if "," in text:
        path, label = text.rsplit(",", 1)
    else:
        path, label = text.rsplit(None, 1)
    return path.strip(), int(label.strip())


def read_split(split_file):
    rows = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parsed = parse_split_line(line)
            if parsed is None:
                continue
            path, label = parsed
            rows.append({"path": path, "label": label, "line_no": line_no})
    return rows


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


def resolve_existing_path(raw_path, base_dir):
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.exists() else None
    resolved = base_dir / raw_path
    return resolved if resolved.exists() else None


def portable_path(raw_path, base_dir):
    resolved = resolve_existing_path(raw_path, base_dir)
    if resolved is None:
        return raw_path
    try:
        rel = resolved.resolve().relative_to(base_dir.resolve())
        return str(rel).replace("/", "\\")
    except ValueError:
        return str(resolved)


def group_samples(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[video_group_key(sample["path"])].append(sample)
    return groups


def group_label(items):
    counts = Counter(item["label"] for item in items)
    if len(counts) > 1:
        return counts.most_common(1)[0][0]
    return next(iter(counts))


def split_group_keys_by_label(groups, seed, train_ratio, val_ratio, test_ratio):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for key, items in groups.items():
        by_label[group_label(items)].append(key)

    buckets = {"train": [], "val": [], "test": [], "holdout": []}
    for label, keys in by_label.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        if n_train + n_val + n_test > n:
            n_test = max(0, n - n_train - n_val)

        train_keys = keys[:n_train]
        val_keys = keys[n_train : n_train + n_val]
        test_keys = keys[n_train + n_val : n_train + n_val + n_test]
        holdout_keys = keys[n_train + n_val + n_test :]

        if not holdout_keys and test_keys:
            holdout_keys.append(test_keys.pop())

        buckets["train"].extend(train_keys)
        buckets["val"].extend(val_keys)
        buckets["test"].extend(test_keys)
        buckets["holdout"].extend(holdout_keys)
    return {name: sorted(keys) for name, keys in buckets.items()}


def write_split(path, samples, base_dir):
    seen = set()
    with open(path, "w", encoding="utf-8", newline="") as f:
        for sample in samples:
            out_path = portable_path(sample["path"], base_dir)
            key = (out_path, sample["label"])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{out_path},{sample['label']}\n")


def summarize_samples(split_name, samples):
    rows = []
    by_source = defaultdict(list)
    for sample in samples:
        by_source[source_name(sample["path"])].append(sample)
    for source, items in sorted(by_source.items()):
        labels = Counter(item["label"] for item in items)
        groups = {video_group_key(item["path"]) for item in items}
        rows.append(
            {
                "split": split_name,
                "source": source,
                "samples": len(items),
                "real": labels.get(0, 0),
                "fake": labels.get(1, 0),
                "groups": len(groups),
            }
        )
    labels = Counter(item["label"] for item in samples)
    groups = {video_group_key(item["path"]) for item in samples}
    rows.append(
        {
            "split": split_name,
            "source": "ALL",
            "samples": len(samples),
            "real": labels.get(0, 0),
            "fake": labels.get(1, 0),
            "groups": len(groups),
        }
    )
    return rows


def overlap_report(split_map):
    rows = []
    names = list(split_map)
    group_sets = {
        name: {video_group_key(sample["path"]) for sample in samples}
        for name, samples in split_map.items()
    }
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = group_sets[left].intersection(group_sets[right])
            rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "left_groups": len(group_sets[left]),
                    "right_groups": len(group_sets[right]),
                    "overlap_groups": len(overlap),
                    "overlap_examples": ";".join(sorted(overlap)[:20]),
                }
            )
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, args, split_paths, summary_rows, overlap_rows):
    lines = []
    lines.append("# Phase 2 Multi-Domain Split V2 Report")
    lines.append("")
    lines.append(f"- Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append(f"- DFDC group ratios: train={args.dfdc_train_ratio}, val={args.dfdc_val_ratio}, test={args.dfdc_test_ratio}, holdout=remaining")
    lines.append("")
    lines.append("## Output Splits")
    lines.append("")
    for name, path_item in split_paths.items():
        lines.append(f"- {name}: `{path_item}`")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("")
    lines.append("| Split | Source | Samples | Real | Fake | Groups |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['source']} | {row['samples']} | {row['real']} | {row['fake']} | {row['groups']} |"
        )
    lines.append("")
    lines.append("## Video Group Overlap")
    lines.append("")
    lines.append("| Left | Right | Overlap Groups |")
    lines.append("|---|---|---:|")
    for row in overlap_rows:
        lines.append(f"| {row['left_split']} | {row['right_split']} | {row['overlap_groups']} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Existing train/val/test samples are preserved and DFDC samples are added by video group.")
    lines.append("- `dfdc_holdout_test.txt` must not be used during training or threshold selection.")
    lines.append("- Old V1 splits are not overwritten.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create candidate V2 multi-domain splits.")
    parser.add_argument("--base-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--train", default="splits/train.txt")
    parser.add_argument("--val", default="splits/val.txt")
    parser.add_argument("--test", default="splits/test.txt")
    parser.add_argument("--dfdc", default="splits/dfdc_test.txt")
    parser.add_argument("--out-dir", default="splits/v2")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dfdc-train-ratio", type=float, default=0.60)
    parser.add_argument("--dfdc-val-ratio", type=float, default=0.20)
    parser.add_argument("--dfdc-test-ratio", type=float, default=0.10)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    original_train = read_split(base_dir / args.train)
    original_val = read_split(base_dir / args.val)
    original_test = read_split(base_dir / args.test)
    dfdc_samples = read_split(base_dir / args.dfdc)

    dfdc_groups = group_samples(dfdc_samples)
    buckets = split_group_keys_by_label(
        dfdc_groups,
        seed=args.seed,
        train_ratio=args.dfdc_train_ratio,
        val_ratio=args.dfdc_val_ratio,
        test_ratio=args.dfdc_test_ratio,
    )

    dfdc_parts = {
        name: [sample for key in keys for sample in dfdc_groups[key]]
        for name, keys in buckets.items()
    }

    split_map = {
        "train_v2": original_train + dfdc_parts["train"],
        "val_v2": original_val + dfdc_parts["val"],
        "test_v2": original_test + dfdc_parts["test"],
        "dfdc_holdout_test": dfdc_parts["holdout"],
        "dfdc_train_part": dfdc_parts["train"],
        "dfdc_val_part": dfdc_parts["val"],
        "dfdc_test_part": dfdc_parts["test"],
    }

    split_paths = {
        "train_v2": out_dir / "train_v2.txt",
        "val_v2": out_dir / "val_v2.txt",
        "test_v2": out_dir / "test_v2.txt",
        "dfdc_holdout_test": out_dir / "dfdc_holdout_test.txt",
        "dfdc_train_part": out_dir / "dfdc_train_part.txt",
        "dfdc_val_part": out_dir / "dfdc_val_part.txt",
        "dfdc_test_part": out_dir / "dfdc_test_part.txt",
    }

    for name, samples in split_map.items():
        write_split(split_paths[name], samples, base_dir)

    summary_rows = []
    for name in ["train_v2", "val_v2", "test_v2", "dfdc_holdout_test", "dfdc_train_part", "dfdc_val_part", "dfdc_test_part"]:
        summary_rows.extend(summarize_samples(name, split_map[name]))

    overlaps = overlap_report({name: split_map[name] for name in ["train_v2", "val_v2", "test_v2", "dfdc_holdout_test"]})

    write_csv(out_dir / "split_v2_summary.csv", summary_rows, ["split", "source", "samples", "real", "fake", "groups"])
    write_csv(out_dir / "split_v2_overlap.csv", overlaps, ["left_split", "right_split", "left_groups", "right_groups", "overlap_groups", "overlap_examples"])
    (out_dir / "split_v2_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "seed": args.seed,
                "dfdc_ratios": {
                    "train": args.dfdc_train_ratio,
                    "val": args.dfdc_val_ratio,
                    "test": args.dfdc_test_ratio,
                    "holdout": 1.0 - args.dfdc_train_ratio - args.dfdc_val_ratio - args.dfdc_test_ratio,
                },
                "split_paths": {name: str(path) for name, path in split_paths.items()},
                "summary": summary_rows,
                "overlap": overlaps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(out_dir / "PHASE2_SPLIT_V2_REPORT.md", args, split_paths, summary_rows, overlaps)

    print(f"Phase 2 V2 split creation complete: {out_dir}")
    print(f"- {out_dir / 'PHASE2_SPLIT_V2_REPORT.md'}")
    print(f"- {out_dir / 'split_v2_summary.csv'}")
    print(f"- {out_dir / 'split_v2_overlap.csv'}")


if __name__ == "__main__":
    main()
