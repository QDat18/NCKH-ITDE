import argparse
import csv
import json
import os
import random
import re
import shutil
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
    return {"path": path.strip(), "label": int(label.strip())}


def read_split(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = parse_split_line(line)
            if item:
                rows.append(item)
    return rows


def write_split(path, rows):
    seen = set()
    with open(path, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            key = (row["path"], row["label"])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{row['path']},{row['label']}\n")


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


def group_samples(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[video_group_key(row["path"])].append(row)
    return groups


def group_label(items):
    labels = Counter(item["label"] for item in items)
    return labels.most_common(1)[0][0]


def group_source(items):
    return source_name(items[0]["path"])


def select_validation_groups(groups, targets, seed):
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for key, items in groups.items():
        by_cell[(group_source(items), group_label(items))].append(key)

    selected = set()
    target_report = []
    for cell, target_samples in targets.items():
        source, label = cell
        candidates = sorted(by_cell.get(cell, []))
        rng.shuffle(candidates)
        selected_samples = 0
        selected_groups = 0
        for key in candidates:
            if selected_samples >= target_samples:
                break
            selected.add(key)
            selected_samples += len(groups[key])
            selected_groups += 1
        target_report.append(
            {
                "source": source,
                "label": label,
                "target_samples": target_samples,
                "selected_samples": selected_samples,
                "selected_groups": selected_groups,
                "available_groups": len(candidates),
            }
        )
    return selected, target_report


def flatten_groups(groups, keys):
    rows = []
    for key in sorted(keys):
        rows.extend(groups[key])
    return rows


def summarize(split_name, rows):
    out = []
    by_source = defaultdict(list)
    for row in rows:
        by_source[source_name(row["path"])].append(row)
    for source, items in sorted(by_source.items()):
        labels = Counter(item["label"] for item in items)
        groups = {video_group_key(item["path"]) for item in items}
        out.append(
            {
                "split": split_name,
                "source": source,
                "samples": len(items),
                "real": labels.get(0, 0),
                "fake": labels.get(1, 0),
                "groups": len(groups),
            }
        )
    labels = Counter(item["label"] for item in rows)
    groups = {video_group_key(item["path"]) for item in rows}
    out.append(
        {
            "split": split_name,
            "source": "ALL",
            "samples": len(rows),
            "real": labels.get(0, 0),
            "fake": labels.get(1, 0),
            "groups": len(groups),
        }
    )
    return out


def overlap_report(split_map):
    rows = []
    group_sets = {
        name: {video_group_key(row["path"]) for row in rows}
        for name, rows in split_map.items()
    }
    names = list(split_map)
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
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, args, split_paths, summary_rows, target_report, overlaps):
    lines = []
    lines.append("# Multi-Domain Split V3 Report")
    lines.append("")
    lines.append(f"- Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append("- Source pool: `train_v2 + val_v2` only.")
    lines.append("- `test_v2` and `dfdc_holdout_test` are copied unchanged and must stay held out.")
    lines.append("")
    lines.append("## Output Splits")
    lines.append("")
    for name, split_path in split_paths.items():
        lines.append(f"- {name}: `{split_path}`")
    lines.append("")
    lines.append("## Validation Targets")
    lines.append("")
    lines.append("| Source | Label | Target samples | Selected samples | Selected groups | Available groups |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in target_report:
        lines.append(
            f"| {row['source']} | {row['label']} | {row['target_samples']} | "
            f"{row['selected_samples']} | {row['selected_groups']} | {row['available_groups']} |"
        )
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
    lines.append("| Left | Right | Overlap groups |")
    lines.append("|---|---|---:|")
    for row in overlaps:
        lines.append(f"| {row['left_split']} | {row['right_split']} | {row['overlap_groups']} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Create V3 train/validation splits with a more representative validation set.")
    parser.add_argument("--train-v2", default="splits/v2/train_v2.txt")
    parser.add_argument("--val-v2", default="splits/v2/val_v2.txt")
    parser.add_argument("--test-v2", default="splits/v2/test_v2.txt")
    parser.add_argument("--holdout-v2", default="splits/v2/dfdc_holdout_test.txt")
    parser.add_argument("--out-dir", default="splits/v3")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--ff-real-val", type=int, default=300)
    parser.add_argument("--ff-fake-val", type=int, default=300)
    parser.add_argument("--celeb-real-val", type=int, default=300)
    parser.add_argument("--celeb-fake-val", type=int, default=300)
    parser.add_argument("--dfdc-real-val", type=int, default=120)
    parser.add_argument("--dfdc-fake-val", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = SCRIPT_DIR
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_v2 = read_split(base_dir / args.train_v2)
    val_v2 = read_split(base_dir / args.val_v2)
    test_v2 = read_split(base_dir / args.test_v2)
    holdout_v2 = read_split(base_dir / args.holdout_v2)

    pool = train_v2 + val_v2
    groups = group_samples(pool)
    targets = {
        ("FaceForensics++", 0): args.ff_real_val,
        ("FaceForensics++", 1): args.ff_fake_val,
        ("Celeb-DF", 0): args.celeb_real_val,
        ("Celeb-DF", 1): args.celeb_fake_val,
        ("DFDC", 0): args.dfdc_real_val,
        ("DFDC", 1): args.dfdc_fake_val,
    }
    val_keys, target_report = select_validation_groups(groups, targets, args.seed)
    all_keys = set(groups)
    train_keys = all_keys.difference(val_keys)

    split_map = {
        "train_v3": flatten_groups(groups, train_keys),
        "val_v3": flatten_groups(groups, val_keys),
        "test_v3": test_v2,
        "dfdc_holdout_test": holdout_v2,
    }

    split_paths = {
        "train_v3": out_dir / "train_v3.txt",
        "val_v3": out_dir / "val_v3.txt",
        "test_v3": out_dir / "test_v3.txt",
        "dfdc_holdout_test": out_dir / "dfdc_holdout_test.txt",
    }
    for name, rows in split_map.items():
        write_split(split_paths[name], rows)

    summary_rows = []
    for name, rows in split_map.items():
        summary_rows.extend(summarize(name, rows))
    overlaps = overlap_report(split_map)

    write_csv(out_dir / "split_v3_summary.csv", summary_rows, ["split", "source", "samples", "real", "fake", "groups"])
    write_csv(
        out_dir / "split_v3_overlap.csv",
        overlaps,
        ["left_split", "right_split", "left_groups", "right_groups", "overlap_groups", "overlap_examples"],
    )
    write_csv(
        out_dir / "split_v3_validation_targets.csv",
        target_report,
        ["source", "label", "target_samples", "selected_samples", "selected_groups", "available_groups"],
    )
    (out_dir / "split_v3_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "seed": args.seed,
                "split_paths": {name: str(path) for name, path in split_paths.items()},
                "validation_targets": target_report,
                "summary": summary_rows,
                "overlap": overlaps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(out_dir / "SPLIT_V3_REPORT.md", args, split_paths, summary_rows, target_report, overlaps)

    print(f"Split V3 creation complete: {out_dir}")
    print(f"- {out_dir / 'SPLIT_V3_REPORT.md'}")
    print(f"- {out_dir / 'split_v3_summary.csv'}")
    print(f"- {out_dir / 'split_v3_overlap.csv'}")


if __name__ == "__main__":
    main()
