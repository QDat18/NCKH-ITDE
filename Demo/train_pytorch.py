import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import os
import time
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from datetime import datetime
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from dataset_pytorch import (
    DeepfakeDataset,
    build_balanced_sampler,
    get_transforms,
    sampler_weight_summary,
)
from model_pytorch import DeepfakeEfficientNet

# --- MASTER CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/best_pytorch_model_b4_v3_ft.pth"
LOG_CSV = "training_log_v3_ft.csv"

class ClassAwareFocalLoss(nn.Module):
    def __init__(self, fake_weight=1.5, real_weight=1.0, gamma=2.0):
        super().__init__()
        self.fake_weight = float(fake_weight)
        self.real_weight = float(real_weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        class_weight = torch.where(
            targets >= 0.5,
            torch.full_like(targets, self.fake_weight),
            torch.full_like(targets, self.real_weight),
        )
        focal_loss = class_weight * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)

def calculate_eer(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

class MetricEarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def plot_confusion_matrix(y_true, y_pred, epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
    plt.close()


def safe_torch_load(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def load_initial_checkpoint(model, checkpoint_path, device, strict=True):
    if not checkpoint_path:
        return None
    if not os.path.exists(checkpoint_path):
        print(f"Warning: init checkpoint not found, training from ImageNet weights: {checkpoint_path}")
        return None
    checkpoint = safe_torch_load(checkpoint_path, device)
    state_dict = extract_state_dict(checkpoint)
    if strict:
        model.load_state_dict(state_dict)
        print(f"Loaded init checkpoint: {checkpoint_path}")
    else:
        result = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded compatible init weights from: {checkpoint_path}")
        print(f"- Missing keys: {len(result.missing_keys)}")
        print(f"- Unexpected keys: {len(result.unexpected_keys)}")
    return checkpoint


def metrics_at_threshold(y_true, y_probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    preds = (y_probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "f2": float(fbeta_score(y_true, preds, beta=2, zero_division=0)),
        "accuracy": float(np.mean(preds == y_true)),
        "preds": preds,
    }


def select_validation_operating_point(y_true, y_probs, target_precision, selection_metric):
    y_probs = np.asarray(y_probs).astype(float)
    thresholds = sorted(set(y_probs.tolist() + [0.5]), reverse=True)
    rows = [metrics_at_threshold(y_true, y_probs, threshold) for threshold in thresholds]

    if selection_metric == "auc":
        row = metrics_at_threshold(y_true, y_probs, 0.5)
        row["selection_score"] = float(roc_auc_score(y_true, y_probs))
        row["selection_status"] = "auc_checkpoint_threshold_0.5"
        return row

    if selection_metric == "f2":
        row = max(rows, key=lambda item: (item["f2"], item["precision"], item["recall"]))
        row["selection_score"] = row["f2"]
        row["selection_status"] = "max_f2"
        return row

    eligible = [row for row in rows if row["precision"] >= target_precision]
    if eligible:
        row = max(eligible, key=lambda item: (item["recall"], item["f2"], item["precision"]))
        row["selection_score"] = row["recall"]
        row["selection_status"] = "precision_floor_met"
        return row

    row = max(rows, key=lambda item: (item["precision"], item["recall"], item["f2"]))
    row["selection_score"] = row["precision"] - target_precision
    row["selection_status"] = "precision_floor_not_met"
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 deepfake detector.")
    parser.add_argument("--train", default="splits/v2/train_v2.txt", help="Training split file.")
    parser.add_argument("--val", default="splits/v2/val_v2.txt", help="Validation split file.")
    parser.add_argument("--model-out", default=MODEL_SAVE_PATH, help="Output checkpoint path.")
    parser.add_argument("--log-csv", default=LOG_CSV, help="Training log CSV path.")
    parser.add_argument(
        "--init-checkpoint",
        default="models/best_pytorch_model_b4_v2_packaged.pth",
        help="Optional checkpoint to fine-tune from. Use an empty string to train from ImageNet weights.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--sampler",
        choices=["none", "label", "source", "source_label", "combined"],
        default="combined",
        help=(
            "Training sampler. label balances Real/Fake; source balances datasets; "
            "source_label balances each dataset-label cell; combined is a gentler source+label balance."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--fake-loss-weight", type=float, default=1.2)
    parser.add_argument("--real-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--target-precision", type=float, default=0.93)
    parser.add_argument(
        "--selection-metric",
        choices=["recall_at_precision", "f2", "auc"],
        default="recall_at_precision",
        help="Metric used to choose the best checkpoint on validation.",
    )
    parser.add_argument("--hard-samples", default="splits/v3/hard_samples_v3.csv", help="Optional hard sample CSV from mine_hard_samples.py.")
    parser.add_argument("--max-sampler-weight", type=float, default=20.0)
    parser.add_argument(
        "--frequency-branch",
        action="store_true",
        help="Enable RGB EfficientNet-B4 + lightweight high-pass/frequency artifact branch.",
    )
    parser.add_argument(
        "--frequency-mode",
        choices=["none", "laplacian", "fft"],
        default="none",
        help="Frequency map type. If --frequency-branch is set with none, laplacian is used.",
    )
    parser.add_argument("--frequency-features", type=int, default=128, help="Feature width of the frequency branch.")
    return parser.parse_args()


def normalize_frequency_args(args):
    if args.frequency_branch and args.frequency_mode == "none":
        args.frequency_mode = "laplacian"
    if not args.frequency_branch:
        args.frequency_mode = "none"
    return args


def resolve_demo_path(path):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def resolve_optional_path(path):
    if not path:
        return ""
    resolved = resolve_demo_path(path)
    if os.path.exists(resolved):
        return resolved
    print(f"Warning: optional file not found, continuing without it: {path}")
    return ""


def print_dataset_distribution(name, dataset):
    info = dataset.describe_distribution()
    print(f"\n{name} distribution:")
    print(f"- Samples: {info['samples']}")
    print(f"- Labels: {info['labels']}")
    print(f"- Sources: {info['sources']}")
    print(f"- Source-label cells: {info['source_labels']}")


def print_sampler_summary(dataset, sampler_mode, max_sampler_weight):
    if sampler_mode == "none":
        print("\nSampler: none (natural imbalanced distribution)")
        return
    print(f"\nSampler mode: {sampler_mode}")
    print("Source-label sampling weights:")
    for row in sampler_weight_summary(dataset, sampler_mode, max_weight=max_sampler_weight):
        print(
            "- "
            f"{row['source']} label={row['label']} samples={row['samples']} "
            f"selected_weight={row['selected_weight']:.4f} "
            f"(label={row['label_weight']:.4f}, source={row['source_weight']:.4f}, "
            f"source_label={row['source_label_weight']:.4f}, combined={row['combined_weight']:.4f}, "
            f"hard={row['avg_hard_weight']:.4f})"
        )


def move_inputs_to_device(batch_inputs, device):
    if isinstance(batch_inputs, (list, tuple)):
        images = batch_inputs[0].to(device, non_blocking=True)
        frequency = batch_inputs[1].to(device, non_blocking=True)
        return images, frequency
    return batch_inputs.to(device, non_blocking=True), None


def forward_model(model, images, frequency=None):
    if frequency is None:
        return model(images)
    return model(images, frequency=frequency)


def save_training_checkpoint(model, path, epoch, threshold, metrics, args, train_split, val_split):
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_name": "efficientnet_b4",
        "model_variant": "efficientnet_b4_frequency" if args.frequency_branch else "efficientnet_b4_rgb",
        "threshold": float(threshold),
        "epoch": int(epoch),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "splits": {
            "train": os.path.normpath(train_split),
            "val": os.path.normpath(val_split),
        },
        "config": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "init_checkpoint": args.init_checkpoint,
            "sampler": args.sampler,
            "hard_samples": args.hard_samples,
            "max_sampler_weight": float(args.max_sampler_weight),
            "num_workers": int(args.num_workers),
            "patience": int(args.patience),
            "loss": (
                "ClassAwareFocalLoss("
                f"fake_weight={args.fake_loss_weight},real_weight={args.real_loss_weight},gamma={args.focal_gamma})"
            ),
            "selection_metric": args.selection_metric,
            "target_precision": float(args.target_precision),
            "frequency_branch": bool(args.frequency_branch),
            "frequency_mode": args.frequency_mode,
            "frequency_features": int(args.frequency_features),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(checkpoint, path)


def save_run_config(path, args, train_split, val_split):
    out_path = os.path.join(os.path.dirname(path) or ".", "run_config.json")
    payload = {
        "model_name": "efficientnet_b4",
        "train_split": os.path.normpath(train_split),
        "val_split": os.path.normpath(val_split),
        "model_out": os.path.normpath(path),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "init_checkpoint": args.init_checkpoint,
        "sampler": args.sampler,
        "hard_samples": args.hard_samples,
        "max_sampler_weight": float(args.max_sampler_weight),
        "num_workers": int(args.num_workers),
        "patience": int(args.patience),
        "fake_loss_weight": float(args.fake_loss_weight),
        "real_loss_weight": float(args.real_loss_weight),
        "focal_gamma": float(args.focal_gamma),
        "selection_metric": args.selection_metric,
        "target_precision": float(args.target_precision),
        "frequency_branch": bool(args.frequency_branch),
        "frequency_mode": args.frequency_mode,
        "frequency_features": int(args.frequency_features),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train():
    args = normalize_frequency_args(parse_args())
    train_split = resolve_demo_path(args.train)
    val_split = resolve_demo_path(args.val)
    model_out = resolve_demo_path(args.model_out)
    log_csv = resolve_demo_path(args.log_csv)
    hard_samples = resolve_optional_path(args.hard_samples)
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
    save_run_config(model_out, args, train_split, val_split)
    
    # Load Dataloaders
    train_ds = DeepfakeDataset(
        train_split,
        transform=get_transforms(is_train=True),
        hard_samples_file=hard_samples or None,
        frequency_mode=args.frequency_mode,
    )
    val_ds = DeepfakeDataset(
        val_split,
        transform=get_transforms(is_train=False),
        frequency_mode=args.frequency_mode,
    )
    print_dataset_distribution("Train", train_ds)
    print_dataset_distribution("Validation", val_ds)
    
    train_sampler = build_balanced_sampler(train_ds, mode=args.sampler, max_weight=args.max_sampler_weight)
    print_sampler_summary(train_ds, args.sampler, args.max_sampler_weight)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model Initialization (Scratch from ImageNet weights)
    model = DeepfakeEfficientNet(
        model_name='efficientnet_b4',
        pretrained=True,
        frequency_branch=args.frequency_branch,
        frequency_features=args.frequency_features,
    ).to(DEVICE)
    init_checkpoint = resolve_optional_path(args.init_checkpoint)
    load_initial_checkpoint(model, init_checkpoint, DEVICE, strict=not args.frequency_branch)
    
    # Optimizer & Scheduler (Expert Choice)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed Precision Scaler
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    criterion = ClassAwareFocalLoss(
        fake_weight=args.fake_loss_weight,
        real_weight=args.real_loss_weight,
        gamma=args.focal_gamma,
    )
    early_stopping = MetricEarlyStopping(patience=args.patience)
    history = []
    best_score = -float("inf")

    print(f"Starting MASTER training on {DEVICE}...")
    print(
        f"Config: BatchSize={args.batch_size}, LR={args.lr}, Optimizer=AdamW, "
        f"Scheduler=CosineAnnealing, Sampler={args.sampler}, Selection={args.selection_metric}, "
        f"TargetPrecision={args.target_precision}, FrequencyBranch={args.frequency_branch}, "
        f"FrequencyMode={args.frequency_mode}"
    )
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_inputs, labels in pbar:
            images, frequency = move_inputs_to_device(batch_inputs, DEVICE)
            targets = labels.to(DEVICE).float().unsqueeze(1)
            
            # Label Smoothing (Expert Choice)
            smooth_targets = targets * 0.9 + 0.05
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = forward_model(model, images, frequency)
                loss = criterion(outputs, smooth_targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            total_train += images.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss /= total_train
        current_lr = optimizer.param_groups[0]['lr']

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images, frequency = move_inputs_to_device(batch_inputs, DEVICE)
                targets = labels.to(DEVICE).float().unsqueeze(1)
                
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = forward_model(model, images, frequency)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)
                
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        val_loss /= val_total
        
        # Calculate Advanced Metrics
        val_auc = roc_auc_score(all_labels, all_probs)
        val_eer, eer_threshold = calculate_eer(all_labels, all_probs)
        operating = select_validation_operating_point(
            all_labels,
            all_probs,
            target_precision=args.target_precision,
            selection_metric=args.selection_metric,
        )
        best_threshold = operating["threshold"]
        preds = operating["preds"]
        val_acc = operating["accuracy"]
        val_precision = operating["precision"]
        val_recall = operating["recall"]
        val_f1 = operating["f1"]
        val_f2 = operating["f2"]
        selection_score = operating["selection_score"]
        selection_status = operating["selection_status"]

        print(f"Epoch {epoch}: LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            "Metrics: "
            f"AUC: {val_auc:.4f}, EER: {val_eer:.4f} (EER Thresh: {eer_threshold:.4f}), "
            f"OpThresh: {best_threshold:.4f}, Precision: {val_precision:.4f}, "
            f"Recall: {val_recall:.4f}, F2: {val_f2:.4f}, "
            f"Score: {selection_score:.4f}, Status: {selection_status}"
        )

        # Update Scheduler
        scheduler.step()

        # Save History
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_eer": val_eer,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_f2": val_f2,
            "threshold": best_threshold,
            "eer_threshold": eer_threshold,
            "selection_score": selection_score,
            "selection_status": selection_status,
            "target_precision": args.target_precision,
        })
        pd.DataFrame(history).to_csv(log_csv, index=False)

        # Plot Confusion Matrix (Every 5 epochs as suggested)
        if epoch % 5 == 0 or epoch == 1:
            plot_confusion_matrix(all_labels, preds, epoch)

        # Model Save: optimize recall/F2 at a precision floor instead of AUC only.
        if selection_score > best_score:
            print(f"Validation score improved ({best_score:.4f} -> {selection_score:.4f}). Saving model...")
            best_score = selection_score
            save_training_checkpoint(
                model=model,
                path=model_out,
                epoch=epoch,
                threshold=best_threshold,
                metrics={
                    "val_auc": val_auc,
                    "val_eer": val_eer,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_f2": val_f2,
                    "selection_score": selection_score,
                    "target_precision": args.target_precision,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                args=args,
                train_split=train_split,
                val_split=val_split,
            )

        # Early Stopping
        early_stopping(selection_score)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break

if __name__ == "__main__":
    train()
