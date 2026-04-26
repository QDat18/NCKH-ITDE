import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet
from sklearn.metrics import roc_curve, auc, accuracy_score
import os

def calculate_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer

def evaluate_subset(model, loader, device, name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            
    if len(all_labels) == 0:
        return None

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    eer = calculate_eer(all_labels, all_preds)
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, binary_preds)
    
    return {
        "Dataset": name,
        "Samples": len(all_labels),
        "AUC": roc_auc,
        "EER": eer,
        "Accuracy": acc
    }

def cross_dataset_test():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/best_pytorch_model_final.pth"
    TEST_FILE = "splits/test.txt"
    
    # 1. Read test file and categorize
    with open(TEST_FILE, "r") as f:
        lines = f.readlines()
        
    ff_lines = [l for l in lines if "FF_" in l]
    celeb_lines = [l for l in lines if "Celeb-" in l]
    
    print(f"Total test samples: {len(lines)}")
    print(f"FF++ samples: {len(ff_lines)}")
    print(f"Celeb-DF samples: {len(celeb_lines)}")
    
    # Create temporary split files for subset evaluation
    os.makedirs("splits/temp_cross", exist_ok=True)
    with open("splits/temp_cross/ff_test.txt", "w") as f: f.writelines(ff_lines)
    with open("splits/temp_cross/celeb_test.txt", "w") as f: f.writelines(celeb_lines)
    
    # 2. Load Model
    model = DeepfakeEfficientNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    
    # 3. Evaluate
    results = []
    
    # Evaluate FF++
    if ff_lines:
        ds_ff = DeepfakeDataset("splits/temp_cross/ff_test.txt", transform=get_transforms(is_train=False))
        dl_ff = DataLoader(ds_ff, batch_size=16, shuffle=False)
        results.append(evaluate_subset(model, dl_ff, DEVICE, "FaceForensics++"))
        
    # Evaluate Celeb-DF
    if celeb_lines:
        ds_celeb = DeepfakeDataset("splits/temp_cross/celeb_test.txt", transform=get_transforms(is_train=False))
        dl_celeb = DataLoader(ds_celeb, batch_size=16, shuffle=False)
        results.append(evaluate_subset(model, dl_celeb, DEVICE, "Celeb-DF"))
        
    # 4. Report
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("CROSS-DATASET EVALUATION REPORT")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    # Calculate Generalization Gap
    if len(results) >= 2:
        gap = abs(results[0]['AUC'] - results[1]['AUC'])
        print(f"Generalization Gap (AUC): {gap:.4f}")
        if gap < 0.05:
            print("Conclusion: Strong Generalization (Model is robust)")
        elif gap < 0.15:
            print("Conclusion: Moderate Generalization (Acceptable)")
        else:
            print("Conclusion: Weak Generalization (Overfitted to one source)")
    
    df.to_csv("cross_dataset_results.csv", index=False)

if __name__ == "__main__":
    cross_dataset_test()
