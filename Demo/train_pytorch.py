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

from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet

# --- MASTER CONFIG ---
BATCH_SIZE = 8
EPOCHS = 25
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/best_pytorch_model_final.pth"
LOG_CSV = "training_log_clean.csv"

# Focal Loss for handling hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

def calculate_eer(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

class AUC_EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auc = None
        self.early_stop = False

    def __call__(self, val_auc):
        if self.best_auc is None:
            self.best_auc = val_auc
        elif val_auc < self.best_auc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_auc = val_auc
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

def train():
    os.makedirs("models", exist_ok=True)
    
    # Load Dataloaders
    train_ds = DeepfakeDataset("splits/train.txt", transform=get_transforms(is_train=True))
    val_ds = DeepfakeDataset("splits/val.txt", transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model Initialization (Scratch from ImageNet weights)
    model = DeepfakeEfficientNet(model_name='efficientnet_b4', pretrained=True).to(DEVICE)
    
    # Optimizer & Scheduler (Expert Choice)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    early_stopping = AUC_EarlyStopping(patience=7)
    history = []
    best_auc = 0.0

    print(f"Starting MASTER training on {DEVICE}...")
    print(f"Config: BatchSize={BATCH_SIZE}, LR={LR}, Optimizer=AdamW, Scheduler=CosineAnnealing")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(DEVICE)
            targets = labels.to(DEVICE).float().unsqueeze(1)
            
            # Label Smoothing (Expert Choice)
            smooth_targets = targets * 0.9 + 0.05
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
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
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images = images.to(DEVICE)
                targets = labels.to(DEVICE).float().unsqueeze(1)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
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
        best_threshold = eer_threshold
        # Binary prediction using EER threshold for accuracy
        preds = np.array(all_probs) > eer_threshold
        val_acc = np.mean(preds == np.array(all_labels))

        print(f"Epoch {epoch}: LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics: AUC: {val_auc:.4f}, EER: {val_eer:.4f} (Thresh: {eer_threshold:.4f}), Acc: {val_acc:.4f}")

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
            "threshold": eer_threshold
        })
        pd.DataFrame(history).to_csv(LOG_CSV, index=False)

        # Plot Confusion Matrix (Every 5 epochs as suggested)
        if epoch % 5 == 0 or epoch == 1:
            plot_confusion_matrix(all_labels, preds, epoch)

        # Model Save (Based on Max AUC as suggested)
        if val_auc > best_auc:
            print(f"Validation AUC Improved ({best_auc:.4f} -> {val_auc:.4f}). Saving model...")
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Early Stopping
        early_stopping(val_auc)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break

if __name__ == "__main__":
    train()
