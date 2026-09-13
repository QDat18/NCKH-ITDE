import argparse
import csv
from pathlib import Path


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    if value in (None, ""):
        return None
    return float(value)


def as_int(value):
    if value in (None, ""):
        return 0
    return int(float(value))


def pct(value):
    if value is None:
        return ""
    return f"{100.0 * float(value):.2f}%"


def signed_pp(new, old):
    if new is None or old is None:
        return ""
    delta = 100.0 * (new - old)
    return f"{delta:+.2f} pp"


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def classify_issue(row):
    dataset = row.get("dataset", "")
    condition = row.get("condition", "")
    source = condition if dataset == "internal_by_source" else ""
    precision = as_float(row.get("precision"))
    recall = as_float(row.get("recall"))
    far = as_float(row.get("far"))
    frr = as_float(row.get("frr"))
    f1 = as_float(row.get("f1"))

    if dataset.startswith("external"):
        return "Lỗi ngoài miền cần theo dõi; ưu tiên giảm FN vì đây là tập holdout."
    if condition == "clean" and dataset == "internal_test":
        return "Mốc lỗi tổng thể của mô hình chính."
    if condition == "jpeg_q30":
        return "Ảnh nén làm tăng FN, mô hình bỏ lọt Fake nhiều hơn clean."
    if condition == "gaussian_blur":
        return "Ảnh mờ làm tăng cả FP và FN, cần augmentation/quality control."
    if condition == "motion_blur":
        return "Mờ chuyển động làm recall giảm, dễ bỏ lọt Fake."
    if condition == "sensor_noise":
        return "Nhiễu camera làm FN tăng, nhưng V2 đã cải thiện rất mạnh so với V1."
    if condition == "low_light":
        return "Ánh sáng yếu vẫn ổn hơn các điều kiện nhiễu/mờ, nhưng FP tăng nhẹ."
    if source == "Celeb-DF":
        return "Nguồn khó nhất: precision thấp, FP cao trên ảnh Real."
    if source == "DFDC":
        return "Nguồn DFDC trong test V2 khá tốt, FP thấp nhưng vẫn còn FN."
    if source == "FaceForensics++":
        return "Nguồn mạnh nhất, nhưng vì số mẫu lớn nên FN tuyệt đối vẫn nhiều."
    if precision is not None and recall is not None:
        if precision < 0.85 and recall >= 0.85:
            return "Thiên về báo Fake, cần giảm FP."
        if recall < 0.85 and precision >= 0.90:
            return "Thiên về bảo thủ, cần giảm FN."
    if far is not None and frr is not None:
        if far > frr:
            return "Lỗi nghiêng về FP."
        if frr > far:
            return "Lỗi nghiêng về FN."
    if f1 is not None:
        return "Cần theo dõi thêm bằng ảnh lỗi mẫu."
    return ""


def summarize_metrics(run_dir):
    rows = read_csv(run_dir / "summary_metrics.csv")
    out_rows = []
    for row in rows:
        dataset = row["dataset"]
        condition = row["condition"]
        if dataset == "internal_test" or dataset == "internal_by_source" or dataset.startswith("external"):
            real = as_int(row["real"])
            fake = as_int(row["fake"])
            fp = as_int(row["fp"])
            fn = as_int(row["fn"])
            tn = as_int(row["tn"])
            tp = as_int(row["tp"])
            out_rows.append(
                {
                    "scope": dataset,
                    "source_or_condition": condition,
                    "samples": as_int(row["samples"]),
                    "real": real,
                    "fake": fake,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "fp_rate_on_real": pct(safe_div(fp, real)),
                    "fn_rate_on_fake": pct(safe_div(fn, fake)),
                    "precision": pct(as_float(row["precision"])),
                    "recall": pct(as_float(row["recall"])),
                    "f1": pct(as_float(row["f1"])),
                    "auc_roc": pct(as_float(row["auc_roc"])),
                    "main_issue": classify_issue(row),
                }
            )
    return out_rows


def summarize_error_examples(prediction_csv, max_each=20):
    rows = read_csv(prediction_csv)
    errors = []
    for row in rows:
        if str(row.get("correct", "")).lower() == "true":
            continue
        label = as_int(row["label"])
        prob = as_float(row["prob_fake"])
        err = row.get("error_type", "")
        confidence_gap = prob if err == "FP" else 1.0 - prob
        errors.append(
            {
                "error_type": err,
                "source": row.get("source", ""),
                "label": "Fake" if label == 1 else "Real",
                "prob_fake": f"{prob:.4f}",
                "severity_score": f"{confidence_gap:.4f}",
                "path": row.get("path", ""),
            }
        )

    fps = sorted([r for r in errors if r["error_type"] == "FP"], key=lambda r: float(r["prob_fake"]), reverse=True)
    fns = sorted([r for r in errors if r["error_type"] == "FN"], key=lambda r: float(r["prob_fake"]))
    return fps[:max_each] + fns[:max_each]


def infer_manipulation(path):
    name = Path(path).name
    if "NeuralTextures" in name:
        return "NeuralTextures"
    if "FaceShifter" in name:
        return "FaceShifter"
    if "Deepfakes" in name:
        return "Deepfakes"
    if "Face2Face" in name:
        return "Face2Face"
    if "FaceSwap" in name:
        return "FaceSwap"
    if "Celeb-synthesis" in name:
        return "Celeb-DF synthesis"
    if "Celeb-real" in name:
        return "Celeb-DF real"
    if "FF_real" in name:
        return "FaceForensics++ real"
    if "data_dfdc_test" in path.replace("/", "\\"):
        return "DFDC frame"
    return "Unknown"


def summarize_error_groups(prediction_csv):
    rows = read_csv(prediction_csv)
    groups = {}
    for row in rows:
        if str(row.get("correct", "")).lower() == "true":
            continue
        err = row.get("error_type", "")
        source = row.get("source", "")
        manipulation = infer_manipulation(row.get("path", ""))
        key = (err, source, manipulation)
        item = groups.setdefault(
            key,
            {
                "error_type": err,
                "source": source,
                "manipulation_or_real_group": manipulation,
                "count": 0,
                "min_prob_fake": 1.0,
                "max_prob_fake": 0.0,
                "avg_prob_fake_sum": 0.0,
            },
        )
        prob = as_float(row.get("prob_fake"))
        item["count"] += 1
        item["min_prob_fake"] = min(item["min_prob_fake"], prob)
        item["max_prob_fake"] = max(item["max_prob_fake"], prob)
        item["avg_prob_fake_sum"] += prob

    out = []
    for item in groups.values():
        count = item["count"]
        out.append(
            {
                "error_type": item["error_type"],
                "source": item["source"],
                "manipulation_or_real_group": item["manipulation_or_real_group"],
                "count": count,
                "avg_prob_fake": f"{item['avg_prob_fake_sum'] / count:.4f}" if count else "",
                "min_prob_fake": f"{item['min_prob_fake']:.4f}" if count else "",
                "max_prob_fake": f"{item['max_prob_fake']:.4f}" if count else "",
            }
        )
    return sorted(out, key=lambda r: (r["error_type"], -r["count"], r["source"], r["manipulation_or_real_group"]))


def build_taxonomy(v2_summary):
    lookup = {(r["scope"], r["source_or_condition"]): r for r in v2_summary}

    def value(scope, key, col):
        return lookup.get((scope, key), {}).get(col, "")

    return [
        {
            "error_group": "Domain shift / lệch nguồn",
            "evidence": (
                f"Celeb-DF có FP={value('internal_by_source', 'Celeb-DF', 'fp')}, "
                f"FP-rate Real={value('internal_by_source', 'Celeb-DF', 'fp_rate_on_real')}, "
                f"Precision={value('internal_by_source', 'Celeb-DF', 'precision')}."
            ),
            "impact": "Ảnh thật từ Celeb-DF dễ bị báo nhầm Fake, làm giảm độ tin cậy khi triển khai đa miền.",
            "suggested_fix": "Bổ sung Celeb-DF Real trong validation, kiểm tra lại sampling theo source-label, và rà thủ công các FP Celeb-DF.",
        },
        {
            "error_group": "Fake khó / artifact yếu",
            "evidence": (
                f"Clean internal FN={value('internal_test', 'clean', 'fn')}, "
                f"FN-rate Fake={value('internal_test', 'clean', 'fn_rate_on_fake')}."
            ),
            "impact": "Mô hình vẫn bỏ lọt một phần ảnh Fake, nhất là khi dấu vết giả mạo mờ hoặc nằm ngoài vùng mô hình chú ý.",
            "suggested_fix": "Ưu tiên phân tích FN thủ công, thêm hard-sample mining có kiểm soát, cân nhắc frequency/high-pass branch.",
        },
        {
            "error_group": "JPEG compression",
            "evidence": (
                f"JPEG Q30 FN={value('internal_test', 'jpeg_q30', 'fn')}, "
                f"Recall={value('internal_test', 'jpeg_q30', 'recall')}, "
                f"F1={value('internal_test', 'jpeg_q30', 'f1')}."
            ),
            "impact": "Nén mạnh làm mất artifact, tăng nguy cơ bỏ lọt deepfake.",
            "suggested_fix": "Tăng JPEG augmentation ở train, nhưng giữ mức nén thực tế để không làm ảnh train quá méo.",
        },
        {
            "error_group": "Blur / motion blur",
            "evidence": (
                f"Gaussian blur FP={value('internal_test', 'gaussian_blur', 'fp')}, FN={value('internal_test', 'gaussian_blur', 'fn')}; "
                f"motion blur FN={value('internal_test', 'motion_blur', 'fn')}."
            ),
            "impact": "Ảnh mờ làm suy yếu chi tiết vùng mặt và làm mô hình vừa bỏ lọt Fake vừa báo nhầm Real.",
            "suggested_fix": "Tăng blur augmentation vừa phải, thêm quality filter để loại face quá mờ nếu dùng trong demo.",
        },
        {
            "error_group": "Sensor noise",
            "evidence": (
                f"Sensor noise FN={value('internal_test', 'sensor_noise', 'fn')}, "
                f"Recall={value('internal_test', 'sensor_noise', 'recall')}, "
                f"F1={value('internal_test', 'sensor_noise', 'f1')}."
            ),
            "impact": "Nhiễu camera vẫn làm tăng FN, dù V2 đã cải thiện rất lớn so với V1.",
            "suggested_fix": "Dùng noise augmentation sát camera thực tế, tránh mức noise quá cực đoan làm lệch phân phối train.",
        },
        {
            "error_group": "Low-light / ánh sáng yếu",
            "evidence": (
                f"Low-light FP={value('internal_test', 'low_light', 'fp')}, "
                f"FN={value('internal_test', 'low_light', 'fn')}, "
                f"F1={value('internal_test', 'low_light', 'f1')}."
            ),
            "impact": "Low-light không làm sụp hiệu năng nhưng có xu hướng tăng FP.",
            "suggested_fix": "Thử RandomGamma/CLAHE nhẹ và kiểm tra lại trên tập validation đa nguồn.",
        },
        {
            "error_group": "Threshold trade-off",
            "evidence": "V3_ft giảm FN internal từ 234 xuống 38 nhưng tăng FP từ 82 lên 300.",
            "impact": "Tăng Recall quá mạnh có thể làm hệ thống báo nhầm Real quá nhiều.",
            "suggested_fix": "Không hạ threshold đơn thuần; dùng validation đa nguồn và chọn threshold theo precision floor.",
        },
    ]


def compare_runs(v2_dir, v3_dir):
    v2_rows = read_csv(v2_dir / "summary_metrics.csv")
    v3_rows = read_csv(v3_dir / "summary_metrics.csv")

    def key(row):
        return row["dataset"], row["condition"]

    v2 = {key(row): row for row in v2_rows}
    v3 = {key(row): row for row in v3_rows}
    out = []
    for k in sorted(set(v2).intersection(v3)):
        dataset, condition = k
        if dataset not in {"internal_test", "internal_by_source", "external_DFDC_holdout"}:
            continue
        r2 = v2[k]
        r3 = v3[k]
        fp2 = as_int(r2["fp"])
        fp3 = as_int(r3["fp"])
        fn2 = as_int(r2["fn"])
        fn3 = as_int(r3["fn"])
        out.append(
            {
                "scope": dataset,
                "source_or_condition": condition,
                "v2_precision": pct(as_float(r2["precision"])),
                "v3ft_precision": pct(as_float(r3["precision"])),
                "precision_change": signed_pp(as_float(r3["precision"]), as_float(r2["precision"])),
                "v2_recall": pct(as_float(r2["recall"])),
                "v3ft_recall": pct(as_float(r3["recall"])),
                "recall_change": signed_pp(as_float(r3["recall"]), as_float(r2["recall"])),
                "v2_fp": fp2,
                "v3ft_fp": fp3,
                "fp_change": fp3 - fp2,
                "v2_fn": fn2,
                "v3ft_fn": fn3,
                "fn_change": fn3 - fn2,
                "reading": compare_reading(dataset, condition, fp2, fp3, fn2, fn3, r2, r3),
            }
        )
    return out


def compare_reading(dataset, condition, fp2, fp3, fn2, fn3, r2, r3):
    if dataset == "internal_test" and condition == "clean":
        return "V3_ft giảm FN mạnh nhưng tăng FP rất lớn; không phù hợp làm mô hình cân bằng."
    if dataset == "internal_test" and condition in {"jpeg_q30", "gaussian_blur", "motion_blur", "sensor_noise", "low_light"}:
        return "V3_ft tăng recall trong stress test nhưng đánh đổi bằng FP cao hơn rõ rệt."
    if dataset == "internal_by_source" and condition == "Celeb-DF":
        return "Rủi ro lớn nhất của V3_ft: báo nhầm Real trên Celeb-DF tăng mạnh."
    if dataset == "internal_by_source" and condition == "DFDC":
        return "V3_ft cải thiện recall DFDC, precision giảm nhẹ."
    if dataset == "internal_by_source" and condition == "FaceForensics++":
        return "V3_ft cải thiện recall nhưng tăng FP trên FF++."
    if dataset.startswith("external"):
        return "V3_ft tốt hơn trên holdout DFDC, nhưng quyết định cuối cần xét cả internal đa nguồn."
    return ""


def md_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def build_report(v2_summary, examples, groups, taxonomy, comparison, out_dir, v2_run, v3_run):
    by_source = [r for r in v2_summary if r["scope"] == "internal_by_source"]
    by_condition = [r for r in v2_summary if r["scope"] == "internal_test"]
    external = [r for r in v2_summary if r["scope"].startswith("external")]
    fp_examples = [r for r in examples if r["error_type"] == "FP"][:10]
    fn_examples = [r for r in examples if r["error_type"] == "FN"][:10]
    top_groups = groups[:12]

    source_cols = [
        "source_or_condition",
        "samples",
        "real",
        "fake",
        "fp",
        "fn",
        "fp_rate_on_real",
        "fn_rate_on_fake",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "main_issue",
    ]
    condition_cols = [
        "source_or_condition",
        "samples",
        "fp",
        "fn",
        "fp_rate_on_real",
        "fn_rate_on_fake",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "main_issue",
    ]
    example_cols = ["error_type", "source", "label", "prob_fake", "severity_score", "path"]
    group_cols = ["error_type", "source", "manipulation_or_real_group", "count", "avg_prob_fake", "min_prob_fake", "max_prob_fake"]
    taxonomy_cols = ["error_group", "evidence", "impact", "suggested_fix"]
    compare_cols = [
        "scope",
        "source_or_condition",
        "v2_precision",
        "v3ft_precision",
        "precision_change",
        "v2_recall",
        "v3ft_recall",
        "recall_change",
        "v2_fp",
        "v3ft_fp",
        "fp_change",
        "v2_fn",
        "v3ft_fn",
        "fn_change",
        "reading",
    ]

    report = f"""# Phân tích lỗi chính thức

## 1. Phạm vi

File này phân tích lỗi của mô hình chính thức **EfficientNet-B4 V2** dựa trên evaluation run:

`{v2_run}`

Đối chiếu thêm với bản **B4-V3_ft** để giải thích vì sao V3_ft chỉ nên xem là thí nghiệm high-recall:

`{v3_run}`

Các bảng được tạo từ `summary_metrics.csv` và `*_predictions.csv`, nên số liệu khớp với pipeline đánh giá hiện tại của project.

## 2. Bảng lỗi theo nguồn dữ liệu của B4-V2

{md_table(by_source, source_cols)}

### Nhận xét theo nguồn

- **Celeb-DF là nguồn khó nhất**: Precision chỉ 71.74%, FP = 52/233 ảnh Real, tức khoảng 22.32% ảnh Real Celeb-DF bị báo nhầm Fake. Đây là điểm yếu chính khi xét khả năng tổng quát theo nguồn.
- **DFDC trong test V2 tương đối ổn**: Precision 97.18%, Recall 88.46%, FP chỉ 4. Tuy nhiên FN = 18 cho thấy vẫn còn một phần Fake DFDC bị bỏ lọt.
- **FaceForensics++ là nguồn mạnh nhất về Precision/F1**: Precision 98.05%, F1 92.42%. Tuy nhiên do số lượng Fake lớn, FN tuyệt đối vẫn là 188, cần tiếp tục giảm bỏ lọt.

## 3. Bảng lỗi theo điều kiện kiểm thử của B4-V2

{md_table(by_condition, condition_cols)}

### Nhận xét theo điều kiện

- Ở điều kiện **clean**, mô hình đạt Precision 95.05%, Recall 87.06%, F1 90.88%; lỗi chính là FN = 234, tức vẫn còn bỏ lọt Fake.
- **JPEG Q30** làm Recall giảm còn 82.08%, FN tăng lên 324. Điều này cho thấy ảnh/video nén mạnh làm mất dấu vết giả mạo.
- **Gaussian blur** và **motion blur** đều làm tăng lỗi, đặc biệt motion blur có FN = 325. Các lỗi này phù hợp với nhận định rằng ảnh mờ làm suy yếu artifact vùng mặt.
- **Sensor noise** ở V2 đã ổn hơn rất nhiều so với V1, nhưng FN vẫn là 304. Đây vẫn là điều kiện cần tiếp tục kiểm thử nếu muốn nâng robustness.
- **Low light** là điều kiện robust tốt nhất trong nhóm stress test, F1 91.90% và Recall 90.10%, nhưng FP tăng lên 108.

## 4. Bảng lỗi external DFDC holdout của B4-V2

{md_table(external, condition_cols)}

### Nhận xét external holdout

- Trên DFDC holdout, B4-V2 đạt Precision 95.56%, Recall 85.43%, F1 90.21% và AUC 93.48%.
- FP chỉ có 6 ảnh Real, nghĩa là khi mô hình báo Fake trên holdout thì khá đáng tin.
- FN = 22 ảnh Fake, tức mô hình vẫn còn bỏ lọt một phần deepfake ngoài miền.
- Đây là bằng chứng tốt hơn V1 vì V1 gặp domain shift rất nặng trên external DFDC cũ.

## 5. Mẫu lỗi cần xem thủ công

### False Positive có xác suất Fake cao

{md_table(fp_examples, example_cols)}

### False Negative có xác suất Fake thấp

{md_table(fn_examples, example_cols)}

### Cách đọc bảng mẫu lỗi

- **FP**: ảnh thật nhưng mô hình dự đoán Fake. Nếu `prob_fake` rất cao, đây là lỗi nghiêm trọng vì mô hình tự tin sai.
- **FN**: ảnh Fake nhưng mô hình dự đoán Real. Nếu `prob_fake` rất thấp, đây là lỗi nghiêm trọng vì mô hình gần như không nhận ra dấu vết giả mạo.
- Các mẫu trên nên được mở thủ công để gắn nhãn nguyên nhân: blur, crop chưa tốt, ánh sáng yếu, mặt quá nhỏ, artifact không rõ, hoặc domain khác train.

## 6. Nhóm lỗi theo source và manipulation

{md_table(top_groups, group_cols)}

### Nhận xét theo nhóm lỗi

- FP tập trung nhiều ở ảnh Real, đặc biệt từ Celeb-DF và một số ảnh Real FaceForensics++ có xác suất Fake rất cao.
- FN của FaceForensics++ xuất hiện nhiều ở các nhóm như NeuralTextures và FaceShifter trong danh sách mẫu lỗi. Đây là nhóm đáng xem kỹ vì artifact có thể tinh vi hoặc không nằm rõ ở vùng crop.
- DFDC có ít FP nhưng vẫn có FN, phù hợp với nhận xét rằng external/domain khác vẫn là thách thức chính.

## 7. Taxonomy lỗi và hướng xử lý

{md_table(taxonomy, taxonomy_cols)}

## 8. Đối chiếu lỗi V2 và V3_ft

{md_table(comparison, compare_cols)}

### Nhận xét V2 so với V3_ft

- V3_ft giảm bỏ lọt Fake rất mạnh: internal FN giảm từ 234 xuống 38.
- Tuy nhiên V3_ft làm FP tăng từ 82 lên 300 trên internal test, tức báo nhầm Real nhiều hơn nhiều.
- Rủi ro lớn nhất nằm ở Celeb-DF: FP tăng từ 52 lên 180, Precision giảm từ 71.74% xuống 46.90%.
- Trên DFDC holdout, V3_ft tốt hơn V2: FN giảm từ 22 xuống 8 và F1 tăng từ 90.21% lên 95.33%.
- Vì vậy, V3_ft có giá trị như hướng high-recall/external-robustness, nhưng chưa nên thay V2 làm mô hình chính nếu yêu cầu cân bằng Real/Fake.

## 9. Kết luận chính thức đưa vào báo cáo

Phân tích lỗi cho thấy mô hình EfficientNet-B4 V2 có xu hướng dự đoán khá tin cậy khi phát hiện Fake, thể hiện qua Precision cao trên internal test và DFDC holdout. Tuy nhiên, mô hình vẫn còn hai hạn chế chính. Thứ nhất, một phần ảnh Fake bị bỏ lọt, đặc biệt trong các điều kiện ảnh bị nén, nhiễu hoặc mờ. Thứ hai, lỗi False Positive tập trung nhiều hơn ở Celeb-DF, cho thấy vẫn tồn tại khác biệt miền dữ liệu giữa các nguồn. Thử nghiệm V3_ft giúp tăng Recall rõ rệt nhưng làm số lượng False Positive tăng mạnh, vì vậy phiên bản V2 được chọn là mô hình chính thức do cân bằng hơn giữa phát hiện Fake và hạn chế báo nhầm Real.

## 10. Định hướng cải thiện tiếp theo

1. Rà soát thủ công các ảnh FP/FN có xác suất sai cao để xác định nguyên nhân lỗi.
2. Bổ sung error taxonomy gồm: blur, motion blur, noise, low-light, crop lỗi, mặt nhỏ/nghiêng, artifact khó quan sát.
3. Tăng dữ liệu/augmentation cho các nhóm lỗi nhiều: JPEG compression, blur, sensor noise.
4. Nếu tiếp tục V3, cần dùng validation có Celeb-DF để tránh mô hình over-predict Fake trên Celeb-DF Real.
5. Chỉ nên triển khai frequency/high-pass branch sau khi đã có taxonomy lỗi rõ ràng, để chứng minh nhánh mới thật sự xử lý artifact còn thiếu.

## 11. File đầu ra

- `error_summary_v2_official.csv`: bảng lỗi tổng hợp của B4-V2.
- `hard_error_examples_v2_official.csv`: danh sách FP/FN nghiêm trọng để xem thủ công.
- `error_groups_v2_official.csv`: thống kê lỗi theo source và nhóm ảnh/manipulation.
- `error_taxonomy_v2_official.csv`: taxonomy lỗi và hướng xử lý.
- `v2_v3ft_error_comparison.csv`: bảng đối chiếu lỗi V2 và V3_ft.
- `ERROR_ANALYSIS_CHINH_THUC.md`: file nhận xét chính thức này.
"""
    (out_dir / "ERROR_ANALYSIS_CHINH_THUC.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create official error analysis tables from evaluation outputs.")
    parser.add_argument("--v2-run", default="evaluation_runs/20260806_094118")
    parser.add_argument("--v3-run", default="evaluation_runs/20260806_135305")
    parser.add_argument("--out-dir", default="phase3_outputs/error_analysis_official")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    v2_dir = (base / args.v2_run).resolve()
    v3_dir = (base / args.v3_run).resolve()
    out_dir = (base / args.out_dir).resolve()

    v2_summary = summarize_metrics(v2_dir)
    examples = summarize_error_examples(v2_dir / "internal_test_clean_predictions.csv", args.max_examples)
    groups = summarize_error_groups(v2_dir / "internal_test_clean_predictions.csv")
    taxonomy = build_taxonomy(v2_summary)
    comparison = compare_runs(v2_dir, v3_dir)

    write_csv(
        out_dir / "error_summary_v2_official.csv",
        v2_summary,
        [
            "scope",
            "source_or_condition",
            "samples",
            "real",
            "fake",
            "tp",
            "fp",
            "tn",
            "fn",
            "fp_rate_on_real",
            "fn_rate_on_fake",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "main_issue",
        ],
    )
    write_csv(
        out_dir / "hard_error_examples_v2_official.csv",
        examples,
        ["error_type", "source", "label", "prob_fake", "severity_score", "path"],
    )
    write_csv(
        out_dir / "error_groups_v2_official.csv",
        groups,
        ["error_type", "source", "manipulation_or_real_group", "count", "avg_prob_fake", "min_prob_fake", "max_prob_fake"],
    )
    write_csv(
        out_dir / "error_taxonomy_v2_official.csv",
        taxonomy,
        ["error_group", "evidence", "impact", "suggested_fix"],
    )
    write_csv(
        out_dir / "v2_v3ft_error_comparison.csv",
        comparison,
        [
            "scope",
            "source_or_condition",
            "v2_precision",
            "v3ft_precision",
            "precision_change",
            "v2_recall",
            "v3ft_recall",
            "recall_change",
            "v2_fp",
            "v3ft_fp",
            "fp_change",
            "v2_fn",
            "v3ft_fn",
            "fn_change",
            "reading",
        ],
    )
    build_report(v2_summary, examples, groups, taxonomy, comparison, out_dir, args.v2_run, args.v3_run)

    print(f"Wrote error analysis to: {out_dir}")
    print(f"- {out_dir / 'ERROR_ANALYSIS_CHINH_THUC.md'}")
    print(f"- {out_dir / 'error_summary_v2_official.csv'}")
    print(f"- {out_dir / 'hard_error_examples_v2_official.csv'}")
    print(f"- {out_dir / 'error_groups_v2_official.csv'}")
    print(f"- {out_dir / 'error_taxonomy_v2_official.csv'}")
    print(f"- {out_dir / 'v2_v3ft_error_comparison.csv'}")


if __name__ == "__main__":
    main()
