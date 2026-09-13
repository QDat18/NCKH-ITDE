from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import auc, confusion_matrix, roc_curve


BASE_DIR = Path(__file__).resolve().parent
TRAIN_LOG = BASE_DIR / "training_log_v2.csv"
EVAL_DIR = BASE_DIR / "evaluation_runs" / "20260806_094118"
PREDICTIONS = EVAL_DIR / "internal_test_clean_predictions.csv"
SUMMARY = EVAL_DIR / "summary_metrics.csv"
OUT_DIR = BASE_DIR / "phase3_outputs" / "chapter3_figures"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_clean_metrics() -> pd.Series:
    metrics = pd.read_csv(SUMMARY)
    row = metrics[(metrics["dataset"] == "internal_test") & (metrics["condition"] == "clean")]
    if row.empty:
        raise ValueError("Cannot find internal_test/clean row in summary_metrics.csv")
    return row.iloc[0]


def plot_training_curves() -> Path:
    log = pd.read_csv(TRAIN_LOG)

    best_auc_idx = log["val_auc"].idxmax()
    best_epoch = int(log.loc[best_auc_idx, "epoch"])
    best_auc = float(log.loc[best_auc_idx, "val_auc"])
    best_acc = float(log.loc[best_auc_idx, "val_acc"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=220)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(log["epoch"], log["val_acc"], marker="o", linewidth=2.0, markersize=3.8, label="Validation Accuracy")
    ax.plot(log["epoch"], log["val_auc"], marker="s", linewidth=2.0, markersize=3.4, label="Validation AUC")
    ax.axvline(best_epoch, color="#6b7280", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.scatter([best_epoch], [best_auc], color="#111827", s=24, zorder=5)
    ax.set_title("Validation Accuracy and AUC", fontsize=11, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.75, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    ax.text(
        0.03,
        0.06,
        f"Best epoch: {best_epoch}\nVal Acc: {pct(best_acc)}\nVal AUC: {pct(best_auc)}",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    ax = axes[1]
    ax.plot(log["epoch"], log["train_loss"], marker="o", linewidth=2.0, markersize=3.8, label="Train Loss")
    ax.plot(log["epoch"], log["val_loss"], marker="s", linewidth=2.0, markersize=3.4, label="Validation Loss")
    ax.axvline(best_epoch, color="#6b7280", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.set_title("Training and Validation Loss", fontsize=11, weight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, fontsize=8, loc="upper right")

    fig.suptitle("EfficientNet-B4 V2 Training Curves", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = OUT_DIR / "figure_13_training_accuracy_loss_b4_v2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_evaluation_result() -> Path:
    metrics = load_clean_metrics()
    preds = pd.read_csv(PREDICTIONS)
    y_true = preds["label"].astype(int).to_numpy()
    y_score = preds["prob_fake"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35), dpi=220, gridspec_kw={"width_ratios": [1.12, 0.88]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.4, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.set_title("ROC Curve on Internal Test V2", fontsize=11, weight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    ax.text(
        0.04,
        0.64,
        "\n".join(
            [
                f"Accuracy:  {pct(metrics['accuracy'])}",
                f"Precision: {pct(metrics['precision'])}",
                f"Recall:    {pct(metrics['recall'])}",
                f"F1-score:  {pct(metrics['f1'])}",
                f"Threshold: {metrics['threshold']:.4f}",
            ]
        ),
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    ax = axes[1]
    names = ["Acc", "Precision", "Recall", "F1", "AUC"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc_roc"],
    ]
    colors = ["#0ea5e9", "#16a34a", "#f59e0b", "#6366f1", "#dc2626"]
    bars = ax.barh(names, values, color=colors, alpha=0.88)
    ax.set_xlim(0.75, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Main Evaluation Metrics", fontsize=11, weight="bold")
    ax.grid(True, axis="x", alpha=0.28)
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.005, 0.995),
            bar.get_y() + bar.get_height() / 2,
            pct(value),
            va="center",
            fontsize=8,
        )

    ax.text(
        0.02,
        -0.22,
        f"Samples: {int(metrics['samples'])} | FP: {int(metrics['fp'])} | FN: {int(metrics['fn'])}",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    fig.suptitle("EfficientNet-B4 V2 Evaluation Result", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = OUT_DIR / "figure_14_model_evaluation_b4_v2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix() -> Path:
    metrics = load_clean_metrics()
    preds = pd.read_csv(PREDICTIONS)
    y_true = preds["label"].astype(int).to_numpy()
    y_pred = preds["pred"].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=220)
    fig.patch.set_facecolor("white")

    im = ax.imshow(cm, cmap="Blues", vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Number of samples", rotation=270, labelpad=16)

    labels = ["Real", "Deepfake"]
    ax.set_xticks([0, 1], labels=[f"Predicted\n{x}" for x in labels])
    ax.set_yticks([0, 1], labels=[f"Actual {x}" for x in labels])
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)
    ax.set_title("Confusion Matrix - EfficientNet-B4 V2", fontsize=13, weight="bold", pad=14)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = cm / row_sums
    max_value = cm.max()
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > max_value * 0.55 else "#111827"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({row_pct[i, j] * 100:.2f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                weight="bold",
            )

    summary = (
        f"Accuracy: {pct(metrics['accuracy'])} | Precision: {pct(metrics['precision'])} | "
        f"Recall: {pct(metrics['recall'])} | F1: {pct(metrics['f1'])}"
    )
    ax.text(
        0.5,
        -0.18,
        summary,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([-.5, .5, 1.5], minor=True)
    ax.set_yticks([-.5, .5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    out = OUT_DIR / "figure_15_confusion_matrix_b4_v2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_caption_file(training_fig: Path, evaluation_fig: Path, confusion_fig: Path) -> Path:
    metrics = load_clean_metrics()
    out = OUT_DIR / "chapter3_figure_captions.md"
    text = f"""# Goi y chen hinh Chuong III

## Hinh 13

Chen file:

`{training_fig}`

Ten hinh:

**Hinh 13. Bieu do Accuracy/AUC va Loss trong qua trinh huan luyen mo hinh EfficientNet-B4 V2**

Nguon: Nhom nghien cuu, tong hop tu file `training_log_v2.csv`.

Luu y khi viet bao cao: file log V2 khong luu train accuracy, vi vay bieu do ben trai trinh bay Validation Accuracy va Validation AUC; bieu do ben phai trinh bay Train Loss va Validation Loss.

## Hinh 14

Chen file:

`{evaluation_fig}`

Ten hinh:

**Hinh 14. Ket qua danh gia mo hinh EfficientNet-B4 V2 tren tap kiem thu noi bo V2**

Nguon: Nhom nghien cuu, tong hop tu evaluation run `20260806_094118`.

Doan dien giai ngan:

Tren tap kiem thu noi bo V2 gom {int(metrics['samples']):,} anh, mo hinh EfficientNet-B4 V2 dat Accuracy {pct(metrics['accuracy'])}, Precision {pct(metrics['precision'])}, Recall {pct(metrics['recall'])}, F1-score {pct(metrics['f1'])} va ROC-AUC {pct(metrics['auc_roc'])}. So loi False Positive la {int(metrics['fp'])}, trong khi False Negative la {int(metrics['fn'])}. Ket qua cho thay mo hinh co kha nang phan biet tot giua anh that va anh Deepfake, dac biet Precision cao cho thay khi mo hinh du doan Fake thi xac suat du doan dung tuong doi cao. Tuy nhien, Recall {pct(metrics['recall'])} cho thay mo hinh van con bo lot mot phan anh Deepfake.

## Hinh 15

Chen file:

`{confusion_fig}`

Ten hinh:

**Hinh 15. Ma tran nham lan cua mo hinh EfficientNet-B4 V2 tren tap kiem thu noi bo V2**

Nguon: Nhom nghien cuu, tong hop tu evaluation run `20260806_094118`.

Doan dien giai ngan:

Ma tran nham lan cho thay mo hinh phan loai dung {int(metrics['tn'])} anh Real va {int(metrics['tp'])} anh Deepfake. Tuy nhien, mo hinh van ghi nhan {int(metrics['fp'])} truong hop False Positive, tuc anh Real bi du doan nham thanh Deepfake, va {int(metrics['fn'])} truong hop False Negative, tuc anh Deepfake bi bo lot thanh Real. Ket qua nay phu hop voi xu huong Precision cao nhung Recall chua dat muc tuyet doi cua mo hinh.
"""
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training_fig = plot_training_curves()
    evaluation_fig = plot_evaluation_result()
    confusion_fig = plot_confusion_matrix()
    captions = write_caption_file(training_fig, evaluation_fig, confusion_fig)
    print(f"Created: {training_fig}")
    print(f"Created: {evaluation_fig}")
    print(f"Created: {confusion_fig}")
    print(f"Created: {captions}")


if __name__ == "__main__":
    main()
