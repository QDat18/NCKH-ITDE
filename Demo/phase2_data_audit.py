import argparse
import csv
import json
import os
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


def summarize_split(split_name, split_path, base_dir):
    samples = read_split(split_path)
    summary = {}
    missing = []
    groups_by_source = defaultdict(set)
    identities_by_source = defaultdict(set)

    for sample in samples:
        raw_path = sample["path"]
        label = sample["label"]
        source = source_name(raw_path)
        key = (split_name, source)
        if key not in summary:
            summary[key] = {
                "split": split_name,
                "source": source,
                "samples": 0,
                "real": 0,
                "fake": 0,
                "existing": 0,
                "missing": 0,
                "groups": 0,
                "identity_ids": 0,
            }
        row = summary[key]
        row["samples"] += 1
        row["real" if label == 0 else "fake"] += 1
        resolved, exists = resolve_existing_path(raw_path, base_dir)
        if exists:
            row["existing"] += 1
        else:
            row["missing"] += 1
            missing.append(
                {
                    "split": split_name,
                    "line_no": sample["line_no"],
                    "path": raw_path,
                    "resolved_path": str(resolved),
                    "label": label,
                    "source": source,
                }
            )
        groups_by_source[key].add(video_group_key(raw_path))
        identities_by_source[key].update(identity_ids(raw_path))

    labels = Counter(sample["label"] for sample in samples)
    all_key = (split_name, "ALL")
    summary[all_key] = {
        "split": split_name,
        "source": "ALL",
        "samples": len(samples),
        "real": labels.get(0, 0),
        "fake": labels.get(1, 0),
        "existing": len(samples) - len(missing),
        "missing": len(missing),
        "groups": len({video_group_key(sample["path"]) for sample in samples}),
        "identity_ids": len(set().union(*(identity_ids(sample["path"]) for sample in samples))) if samples else 0,
    }

    for key, groups in groups_by_source.items():
        summary[key]["groups"] = len(groups)
        summary[key]["identity_ids"] = len(identities_by_source[key])

    return samples, list(summary.values()), missing


def overlap_rows(split_samples):
    rows = []
    names = list(split_samples)
    group_sets = {
        name: {video_group_key(sample["path"]) for sample in samples}
        for name, samples in split_samples.items()
    }
    id_sets = {
        name: set().union(*(identity_ids(sample["path"]) for sample in samples)) if samples else set()
        for name, samples in split_samples.items()
    }
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


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, summary_rows, overlaps, missing_rows):
    lines = []
    lines.append("# Phase 2 Data Audit Report")
    lines.append("")
    lines.append(f"- Created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("- Purpose: audit current split composition, missing files, and leakage risk before V2 multi-domain split.")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("")
    lines.append("| Split | Source | Samples | Real | Fake | Existing | Missing | Groups | Identity IDs |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['split']} | {row['source']} | {row['samples']} | {row['real']} | {row['fake']} | "
            f"{row['existing']} | {row['missing']} | {row['groups']} | {row['identity_ids']} |"
        )
    lines.append("")
    lines.append("## Split Overlap")
    lines.append("")
    lines.append("| Left | Right | Overlap Groups | Overlap Identity IDs |")
    lines.append("|---|---|---:|---:|")
    for row in overlaps:
        lines.append(
            f"| {row['left_split']} | {row['right_split']} | {row['overlap_groups']} | {row['overlap_identity_ids']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if missing_rows:
        lines.append(f"- Missing file paths found: {len(missing_rows)}. See `missing_files.csv`.")
    else:
        lines.append("- No missing file paths found in the audited splits.")
    risky = [row for row in overlaps if row["overlap_groups"] or row["overlap_identity_ids"]]
    if risky:
        lines.append("- Some split overlaps were detected. See `split_overlap.csv` before training V2.")
    else:
        lines.append("- No video-group or identity overlap detected among audited splits.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Audit current deepfake data splits for Phase 2.")
    parser.add_argument("--base-dir", default=str(SCRIPT_DIR), help="Base directory for resolving relative paths.")
    parser.add_argument("--train", default="splits/train.txt")
    parser.add_argument("--val", default="splits/val.txt")
    parser.add_argument("--test", default="splits/test.txt")
    parser.add_argument("--dfdc", default="splits/dfdc_test.txt")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = base_dir / out_dir
    else:
        out_dir = base_dir / "phase2_outputs" / ("data_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": base_dir / args.train,
        "val": base_dir / args.val,
        "test": base_dir / args.test,
        "dfdc_external_current": base_dir / args.dfdc,
    }

    split_samples = {}
    summary_rows = []
    missing_rows = []
    for name, split_path in splits.items():
        if not split_path.exists():
            continue
        samples, summary, missing = summarize_split(name, split_path, base_dir)
        split_samples[name] = samples
        summary_rows.extend(summary)
        missing_rows.extend(missing)

    source_order = {"train": 0, "val": 1, "test": 2, "dfdc_external_current": 3}
    summary_rows.sort(key=lambda r: (source_order.get(r["split"], 99), r["source"] == "ALL", r["source"]))
    overlaps = overlap_rows(split_samples)

    write_csv(
        out_dir / "split_summary.csv",
        summary_rows,
        ["split", "source", "samples", "real", "fake", "existing", "missing", "groups", "identity_ids"],
    )
    write_csv(
        out_dir / "missing_files.csv",
        missing_rows,
        ["split", "line_no", "path", "resolved_path", "label", "source"],
    )
    write_csv(
        out_dir / "split_overlap.csv",
        overlaps,
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
    (out_dir / "phase2_data_audit.json").write_text(
        json.dumps({"summary": summary_rows, "overlap": overlaps, "missing_files": missing_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out_dir / "PHASE2_DATA_AUDIT.md", summary_rows, overlaps, missing_rows)

    print(f"Phase 2 data audit complete: {out_dir}")
    print(f"- {out_dir / 'PHASE2_DATA_AUDIT.md'}")
    print(f"- {out_dir / 'split_summary.csv'}")
    print(f"- {out_dir / 'split_overlap.csv'}")


if __name__ == "__main__":
    main()
