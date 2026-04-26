import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from torch.utils.data import DataLoader
from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet
import os

def generate_roc():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/best_pytorch_model_final.pth"
    BATCH_SIZE = 16

    # 1. Load Data
    test_ds = DeepfakeDataset("splits/test.txt", transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = DeepfakeEfficientNet().to(DEVICE)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        threshold = checkpoint.get("threshold", 0.5)
        print(f"Using Optimal EER Threshold: {threshold:.4f}")
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5
        print("Warning: Defaulting to 0.5 threshold")
    
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating for ROC Curve...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    # 3. Calculate Metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision, Recall, F1 at Optimal EER threshold
    binary_preds = [1 if p > threshold else 0 for p in all_preds]
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, binary_preds, average='binary')

    # 4. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Add metrics text to plot
    metrics_text = f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\nROC AUC Score: {roc_auc:.2f}"
    plt.text(0.05, 0.95, metrics_text, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    save_path = "roc_curve.png"
    plt.savefig(save_path)
    print(f"Saved ROC Curve and metrics to {save_path}")
    
    return precision, recall, f1, roc_auc

if __name__ == "__main__":
    p, r, f, a = generate_roc()
    print("-" * 30)
    print(f"ACTUAL METRICS (TEST SET):")
    print(f"Precision: {p:.2f}")
    print(f"Recall:    {r:.2f}")
    print(f"F1 Score:  {f:.2f}")
    print(f"AUC Score: {a:.2f}")
    print("-" * 30)
