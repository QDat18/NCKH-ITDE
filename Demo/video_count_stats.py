from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LABEL_NAMES = {0: "Real", 1: "Fake"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_split_line(line: str):
    text = line.strip()
    if not text:
        return None
    if "," in text:
        path, label = text.rsplit(",", 1)
    else:
        path, label = text.rsplit(None, 1)
    return path.strip(), int(label.strip())


def source_name(path: str) -> str:
    text = str(path)
    lower = text.lower()
    base = os.path.basename(text)
    if "FF_" in base:
        return "FaceForensics++"
    if "Celeb-" in base:
        return "Celeb-DF"
    if "dfdc" in lower or "deepfake-detection-challenge" in lower:
        return "DFDC"
    return "Other"


def video_group_key(path: str) -> str:
    base = os.path.basename(str(path))
    stem = re.sub(r"\.[^.]+$", "", base)
    stem = re.sub(r"_f\d+$", "", stem)
    return f"{source_name(path)}::{stem}"


def raw_source_name(path: Path) -> str:
    text = str(path).lower()
    if "celeb-real" in text or "celeb-synthesis" in text or "youtube-real" in text or "self-real" in text:
        return "Celeb-DF"
    if "faceforensics" in text or "ff++" in text:
        return "FaceForensics++"
    if "dfdc" in text or "deepfake-detection-challenge" in text:
        return "DFDC"
    return "Other"


def raw_label(path: Path) -> int:
    text = str(path).lower()
    if "synthesis" in text or "fake" in text or "manipulated" in text:
        return 1
    return 0


def read_split(path: Path, split_name: str):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parsed = parse_split_line(line)
            if parsed is None:
                continue
            raw_path, label = parsed
            rows.append(
                {
                    "split": split_name,
                    "line_no": line_no,
                    "path": raw_path,
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, str(label)),
                    "source": source_name(raw_path),
                    "group": video_group_key(raw_path),
                }
            )
    return rows


def summarize_split_videos(rows):
    by_split_source = defaultdict(lambda: {"frames": Counter(), "groups": defaultdict(set)})
    by_split = defaultdict(lambda: {"frames": Counter(), "groups": defaultdict(set)})

    for row in rows:
        split = row["split"]
        source = row["source"]
        label = row["label"]
        group = row["group"]

        by_split_source[(split, source)]["frames"][label] += 1
        by_split_source[(split, source)]["groups"][label].add(group)
        by_split[split]["frames"][label] += 1
        by_split[split]["groups"][label].add(group)

    out_rows = []
    for (split, source), item in sorted(by_split_source.items()):
        out_rows.append(make_split_summary_row(split, source, item))
    for split, item in sorted(by_split.items()):
        out_rows.append(make_split_summary_row(split, "ALL", item))

    split_order = {
        "train_v2": 0,
        "val_v2": 1,
        "test_v2": 2,
        "dfdc_holdout_test": 3,
    }
    source_order = {
        "FaceForensics++": 0,
        "Celeb-DF": 1,
        "DFDC": 2,
        "ALL": 9,
    }
    out_rows.sort(key=lambda r: (split_order.get(r["split"], 99), source_order.get(r["source"], 8), r["source"]))
    return out_rows


def make_split_summary_row(split: str, source: str, item: dict):
    real_frames = item["frames"][0]
    fake_frames = item["frames"][1]
    real_videos = len(item["groups"][0])
    fake_videos = len(item["groups"][1])
    total_frames = real_frames + fake_frames
    total_videos = real_videos + fake_videos
    return {
        "split": split,
        "source": source,
        "real_videos": real_videos,
        "fake_videos": fake_videos,
        "total_videos": total_videos,
        "real_frames": real_frames,
        "fake_frames": fake_frames,
        "total_frames": total_frames,
        "avg_frames_per_video": round(total_frames / total_videos, 2) if total_videos else 0,
    }


def summarize_raw_videos(raw_dir: Path):
    rows = []
    if not raw_dir.exists():
        return rows

    by_folder = defaultdict(lambda: {"count": 0, "bytes": 0})
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        source = raw_source_name(path)
        label = raw_label(path)
        key = (source, path.parent.name, label)
        by_folder[key]["count"] += 1
        by_folder[key]["bytes"] += path.stat().st_size

    for (source, folder, label), item in sorted(by_folder.items()):
        rows.append(
            {
                "source": source,
                "raw_folder": folder,
                "label": label,
                "label_name": LABEL_NAMES.get(label, str(label)),
                "raw_videos": item["count"],
                "size_mb": round(item["bytes"] / (1024 * 1024), 2),
            }
        )
    return rows


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt_int(value) -> str:
    return f"{int(value):,}"


def write_markdown(path: Path, raw_rows, split_rows, split_files):
    lines = []
    lines.append("# Video Count Statistics")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "In split-based tables, one video is counted as one video group after frame extraction. "
        "The group key is inferred from the image filename by removing the frame suffix such as _f0, _f1, ..."
    )
    lines.append("")
    lines.append(
        "This split-based video count is the recommended table for the report because it matches the actual "
        "train/validation/test files used by the code."
    )
    lines.append("")

    lines.append("## Raw Videos Available Locally")
    lines.append("")
    if raw_rows:
        lines.append("| Source | Raw folder | Label | Raw videos | Size (MB) |")
        lines.append("|---|---|---:|---:|---:|")
        for row in raw_rows:
            lines.append(
                f"| {row['source']} | {row['raw_folder']} | {row['label_name']} | "
                f"{fmt_int(row['raw_videos'])} | {row['size_mb']} |"
            )
    else:
        lines.append("No raw video files were found.")
    lines.append("")

    lines.append("## Processed Video Groups Used In Split V2")
    lines.append("")
    lines.append("| Split | Source | Real videos | Fake videos | Total videos | Real frames | Fake frames | Total frames |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in split_rows:
        lines.append(
            f"| {row['split']} | {row['source']} | {fmt_int(row['real_videos'])} | "
            f"{fmt_int(row['fake_videos'])} | {fmt_int(row['total_videos'])} | "
            f"{fmt_int(row['real_frames'])} | {fmt_int(row['fake_frames'])} | {fmt_int(row['total_frames'])} |"
        )
    lines.append("")

    totals = [row for row in split_rows if row["source"] == "ALL"]
    lines.append("## Word-Ready Summary Table")
    lines.append("")
    lines.append("| Dataset split | Real videos | Fake videos | Total videos | Real images | Fake images | Total images |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in totals:
        lines.append(
            f"| {row['split']} | {fmt_int(row['real_videos'])} | {fmt_int(row['fake_videos'])} | "
            f"{fmt_int(row['total_videos'])} | {fmt_int(row['real_frames'])} | "
            f"{fmt_int(row['fake_frames'])} | {fmt_int(row['total_frames'])} |"
        )
    lines.append("")

    lines.append("## Source Files")
    lines.append("")
    for name, split_path in split_files.items():
        lines.append(f"- {name}: `{split_path}`")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def raw_count(raw_rows, folder_name: str, label: int | None = None) -> int:
    total = 0
    for row in raw_rows:
        if row["raw_folder"] != folder_name:
            continue
        if label is not None and row["label"] != label:
            continue
        total += int(row["raw_videos"])
    return total


def write_console_style_report(path: Path, raw_rows):
    celeb_real = raw_count(raw_rows, "Celeb-real", 0)
    youtube_real = raw_count(raw_rows, "YouTube-real", 0)
    celeb_fake = raw_count(raw_rows, "Celeb-synthesis", 1)
    total_real = celeb_real + youtube_real

    width = 58
    lines = [
        "=" * width,
        "THỐNG KÊ CHI TIẾT TẬP DỮ LIỆU DEEPFAKE",
        "=" * width,
        "",
        "[1] DỮ LIỆU THÔ (VIDEO)",
        f"- Celeb-DF Real: {celeb_real} videos",
        f"- YouTube Real: {youtube_real} videos",
        f"- Celeb-DF Fake: {celeb_fake} videos",
        f"=> Tổng Real: {total_real} | Tổng Fake: {celeb_fake}",
        "",
        "Nguồn: Nhóm nghiên cứu thống kê từ thư mục data_raw.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return lines


def aggregate_split_sources(split_rows, include_holdout: bool):
    selected = []
    for row in split_rows:
        if row["source"] == "ALL":
            continue
        if not include_holdout and row["split"] == "dfdc_holdout_test":
            continue
        selected.append(row)

    totals = defaultdict(lambda: Counter())
    for row in selected:
        source = row["source"]
        for field in [
            "real_videos",
            "fake_videos",
            "total_videos",
            "real_frames",
            "fake_frames",
            "total_frames",
        ]:
            totals[source][field] += int(row[field])

    source_order = ["FaceForensics++", "Celeb-DF", "DFDC"]
    return [(source, totals[source]) for source in source_order if source in totals]


def split_total(split_rows, split_name: str):
    for row in split_rows:
        if row["split"] == split_name and row["source"] == "ALL":
            return row
    return None


def write_full_console_report(path: Path, raw_rows, split_rows):
    celeb_real = raw_count(raw_rows, "Celeb-real", 0)
    youtube_real = raw_count(raw_rows, "YouTube-real", 0)
    celeb_fake = raw_count(raw_rows, "Celeb-synthesis", 1)
    self_real = raw_count(raw_rows, "Self-real", 0)
    total_real = celeb_real + youtube_real

    train = split_total(split_rows, "train_v2")
    val = split_total(split_rows, "val_v2")
    test = split_total(split_rows, "test_v2")
    holdout = split_total(split_rows, "dfdc_holdout_test")

    width = 72
    lines = [
        "=" * width,
        "THỐNG KÊ CHI TIẾT TẬP DỮ LIỆU DEEPFAKE",
        "=" * width,
        "",
        "[1] DỮ LIỆU THÔ CÓ TRONG THƯ MỤC data_raw (VIDEO)",
        f"- Celeb-DF Real: {celeb_real} videos",
        f"- YouTube Real: {youtube_real} videos",
        f"- Celeb-DF Fake: {celeb_fake} videos",
        f"=> Tổng Real: {total_real} | Tổng Fake: {celeb_fake}",
    ]
    if self_real:
        lines.append(f"- Ghi chú: Self-real local có thêm {self_real} videos, không tính vào tổng chuẩn Celeb-DF ở trên.")

    lines.extend(
        [
            "",
            "[2] DỮ LIỆU ĐA NGUỒN ĐÃ ĐI VÀO SPLIT V2 (VIDEO GROUP / ẢNH)",
            "- Cách đếm video: gom các frame cùng gốc video bằng cách bỏ hậu tố _f0, _f1, ...",
            "",
        ]
    )

    for source, item in aggregate_split_sources(split_rows, include_holdout=False):
        lines.append(
            f"- {source}: {item['total_videos']} video groups "
            f"({item['real_videos']} Real | {item['fake_videos']} Fake), "
            f"{item['total_frames']} ảnh ({item['real_frames']} Real | {item['fake_frames']} Fake)"
        )

    lines.extend(["", "[3] CHIA TẬP TRAIN / VALIDATION / TEST"])
    for label, row in [("Train", train), ("Validation", val), ("Test nội bộ", test), ("DFDC holdout", holdout)]:
        if not row:
            continue
        lines.append(
            f"- {label}: {row['total_videos']} video groups, {row['total_frames']} ảnh "
            f"({row['real_frames']} Real | {row['fake_frames']} Fake)"
        )

    lines.extend(
        [
            "",
            "Nguồn: Nhóm nghiên cứu thống kê từ data_raw và các file split V2.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return lines


def fmt_vi_int(value) -> str:
    return f"{int(value):,}".replace(",", ".")


def write_processed_split_console_report(path: Path, split_rows):
    train = split_total(split_rows, "train_v2")
    val = split_total(split_rows, "val_v2")
    test = split_total(split_rows, "test_v2")
    holdout = split_total(split_rows, "dfdc_holdout_test")

    width = 76
    lines = [
        "=" * width,
        "THỐNG KÊ DỮ LIỆU SAU TRÍCH XUẤT VÀ CHIA TẬP V2",
        "=" * width,
        "",
        "[2] DỮ LIỆU SAU TRÍCH XUẤT (CROP FACES)",
    ]
    for label, row in [("Train", train), ("Validation", val), ("Test nội bộ", test), ("DFDC holdout", holdout)]:
        if not row:
            continue
        lines.append(
            f"- {label}: {fmt_vi_int(row['total_frames'])} ảnh "
            f"({fmt_vi_int(row['real_frames'])} Real | {fmt_vi_int(row['fake_frames'])} Fake)"
        )
    lines.extend(
        [
            "=> Dữ liệu còn lệch lớp, nên huấn luyện sử dụng sampler cân bằng theo nhãn/nguồn.",
            "",
            "Nguồn: Nhóm nghiên cứu thống kê từ các file split V2.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return lines


def try_write_console_style_png(path: Path, lines):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False

    font = None
    font_candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            font = ImageFont.truetype(font_path, 22)
            break
    if font is None:
        font = ImageFont.load_default()

    padding_x = 28
    padding_y = 22
    line_gap = 8
    line_heights = []
    line_widths = []
    probe = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(probe)
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        line_widths.append(box[2] - box[0])
        line_heights.append(box[3] - box[1])
    width = max(line_widths) + 2 * padding_x
    height = sum(line_heights) + line_gap * (len(lines) - 1) + 2 * padding_y

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = padding_y
    for line, h in zip(lines, line_heights):
        draw.text((padding_x, y), line, fill=(20, 20, 20), font=font)
        y += h + line_gap
    img.save(path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Count raw videos and processed video groups for split V2.")
    parser.add_argument("--base-dir", default=str(SCRIPT_DIR), help="Project Demo directory.")
    parser.add_argument("--raw-dir", default="data_raw", help="Raw video directory relative to base-dir.")
    parser.add_argument("--out-dir", default="phase3_outputs/video_count_stats", help="Output directory relative to base-dir.")
    parser.add_argument("--train", default="splits/v2/train_v2.txt")
    parser.add_argument("--val", default="splits/v2/val_v2.txt")
    parser.add_argument("--test", default="splits/v2/test_v2.txt")
    parser.add_argument("--holdout", default="splits/v2/dfdc_holdout_test.txt")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    raw_dir = base_dir / args.raw_dir
    out_dir = base_dir / args.out_dir
    split_files = {
        "train_v2": base_dir / args.train,
        "val_v2": base_dir / args.val,
        "test_v2": base_dir / args.test,
        "dfdc_holdout_test": base_dir / args.holdout,
    }

    all_rows = []
    for split_name, split_path in split_files.items():
        if split_path.exists():
            all_rows.extend(read_split(split_path, split_name))
        else:
            print(f"Warning: missing split file: {split_path}")

    raw_rows = summarize_raw_videos(raw_dir)
    split_rows = summarize_split_videos(all_rows)

    write_csv(
        out_dir / "raw_video_counts.csv",
        raw_rows,
        ["source", "raw_folder", "label", "label_name", "raw_videos", "size_mb"],
    )
    write_csv(
        out_dir / "split_v2_video_group_counts.csv",
        split_rows,
        [
            "split",
            "source",
            "real_videos",
            "fake_videos",
            "total_videos",
            "real_frames",
            "fake_frames",
            "total_frames",
            "avg_frames_per_video",
        ],
    )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Split video counts are inferred from processed frame filenames by removing the _fN suffix.",
        "raw_video_counts": raw_rows,
        "split_v2_video_group_counts": split_rows,
        "split_files": {name: str(path) for name, path in split_files.items()},
    }
    (out_dir / "video_count_stats.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "VIDEO_COUNT_STATS.md", raw_rows, split_rows, split_files)
    console_lines = write_console_style_report(out_dir / "raw_video_console_report.txt", raw_rows)
    wrote_png = try_write_console_style_png(out_dir / "raw_video_console_report.png", console_lines)
    full_console_lines = write_full_console_report(out_dir / "full_dataset_console_report.txt", raw_rows, split_rows)
    wrote_full_png = try_write_console_style_png(out_dir / "full_dataset_console_report.png", full_console_lines)
    processed_lines = write_processed_split_console_report(out_dir / "processed_split_v2_console_report.txt", split_rows)
    wrote_processed_png = try_write_console_style_png(out_dir / "processed_split_v2_console_report.png", processed_lines)

    print(f"Created: {out_dir / 'raw_video_counts.csv'}")
    print(f"Created: {out_dir / 'split_v2_video_group_counts.csv'}")
    print(f"Created: {out_dir / 'video_count_stats.json'}")
    print(f"Created: {out_dir / 'VIDEO_COUNT_STATS.md'}")
    print(f"Created: {out_dir / 'raw_video_console_report.txt'}")
    if wrote_png:
        print(f"Created: {out_dir / 'raw_video_console_report.png'}")
    else:
        print("Skipped PNG rendering: Pillow is not available.")
    print(f"Created: {out_dir / 'full_dataset_console_report.txt'}")
    if wrote_full_png:
        print(f"Created: {out_dir / 'full_dataset_console_report.png'}")
    else:
        print("Skipped full PNG rendering: Pillow is not available.")
    print(f"Created: {out_dir / 'processed_split_v2_console_report.txt'}")
    if wrote_processed_png:
        print(f"Created: {out_dir / 'processed_split_v2_console_report.png'}")
    else:
        print("Skipped processed split PNG rendering: Pillow is not available.")


if __name__ == "__main__":
    main()
