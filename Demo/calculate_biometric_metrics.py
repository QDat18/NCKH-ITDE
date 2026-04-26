import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from torch.utils.data import DataLoader
from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet
import os

def calculate_eer(fpr, tpr, thresholds):
    """
    Finds the Equal Error Rate (EER) where FPR = FRR (1 - TPR).
    """
    fnr = 1 - tpr
    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    return eer, eer_threshold

def generate_biometric_report():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the tuned model for the most accurate personalization metrics
    MODEL_PATH = "models/best_pytorch_model_final.pth"

    BATCH_SIZE = 16

    # 1. Load Data
    test_ds = DeepfakeDataset("splits/test.txt", transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load model with support for repackaged checkpoints
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = DeepfakeEfficientNet(pretrained=False).to(DEVICE)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        threshold = checkpoint.get("threshold", 0.5)
        print(f"Using Optimal EER Threshold from model: {threshold:.4f}")
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5
        print("Warning: No threshold found in model, defaulting to 0.5")
        
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Analyzing Biometric Metrics using {MODEL_PATH}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    # 3. Calculate Biometric Metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # EER calculation
    eer, eer_threshold = calculate_eer(fpr, tpr, thresholds)
    
    # FAR, FRR at standard 0.5 threshold
    binary_preds_05 = [1 if p > 0.5 else 0 for p in all_preds]
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, binary_preds_05, average='binary')
    
    # FAR = FPR at 0.5
    # FRR = FNR at 0.5
    far_05 = fpr[np.nanargmin(np.absolute(thresholds - 0.5))]
    frr_05 = 1 - tpr[np.nanargmin(np.absolute(thresholds - 0.5))]

    # 4. Save results to CSV for record
    results_df = pd.DataFrame({
        'Metric': ['AUC', 'EER', 'EER_Threshold', 'FAR (at 0.5)', 'FRR (at 0.5)', 'Precision', 'Recall', 'F1'],
        'Value': [roc_auc, eer, eer_threshold, far_05, frr_05, precision, recall, f1]
    })
    results_df.to_csv("biometric_metrics.csv", index=False)

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Plot EER point
    plt.scatter(eer, 1-eer, color='red', s=100, label=f'EER Point ({eer:.4f})', zorder=5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FAR (False Acceptance Rate)')
    plt.ylabel('1 - FRR (True Positive Rate)')
    plt.title('Biometric Performance Analysis (FAR vs FRR)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Text box with metrics
    report = (f"AUC: {roc_auc:.4f}\n"
              f"EER: {eer:.4f} (at thresh {eer_threshold:.4f})\n"
              f"FAR (0.5): {far_05:.4f}\n"
              f"FRR (0.5): {frr_05:.4f}\n"
              f"F1 Score: {f1:.4f}")
    plt.text(0.55, 0.2, report, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig("biometric_performance.png")
    print("-" * 40)
    print("BIOMETRIC ANALYSIS REPORT:")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Optimal Threshold at EER: {eer_threshold:.4f}")
    print(f"FAR at 0.5 threshold: {far_05*100:.2f}%")
    print(f"FRR at 0.5 threshold: {frr_05*100:.2f}%")
    print(f"AUC Score: {roc_auc:.4f}")
    print("-" * 40)
    print("Plots saved to biometric_performance.png")

if __name__ == "__main__":
    generate_biometric_report()
