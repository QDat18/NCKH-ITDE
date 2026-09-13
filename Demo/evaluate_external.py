import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
import tempfile
import argparse

def calculate_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, thresholds[idx]

def create_temp_split(dataset_dir):
    fd, temp_path = tempfile.mkstemp(suffix='.txt', text=True)
    count_real, count_fake = 0, 0
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    path = os.path.join(root, file)
                    path_lower = path.lower()
                    
                    # Heuristic to infer label
                    if 'real' in path_lower or '0_real' in path_lower or 'original' in path_lower:
                        label = 0
                        count_real += 1
                    elif 'fake' in path_lower or '1_fake' in path_lower or 'manipulated' in path_lower:
                        label = 1
                        count_fake += 1
                    else:
                        # Default to 1 (fake) if we can't tell, but warn the user.
                        # For cross dataset testing usually it's organized in subfolders
                        label = 1
                        count_fake += 1
                    
                    # Convert backward slashes to forward for consistency if needed, but python opens fine
                    f.write(f"{path},{label}\n")
    return temp_path, count_real, count_fake

def main():
    parser = argparse.ArgumentParser(description="Evaluate on external dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the external dataset directory containing 'real' and 'fake' subfolders")
    parser.add_argument("--model", type=str, default="models/best_pytorch_model_final.pth", help="Path to model weights")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    if not os.path.exists(args.dataset_dir):
        print(f"Error: Directory {args.dataset_dir} does not exist.")
        return

    print("Scanning directory for images...")
    split_file, count_real, count_fake = create_temp_split(args.dataset_dir)
    print(f"Found {count_real} REAL and {count_fake} FAKE images.")

    if count_real == 0 and count_fake == 0:
        print("No images found. Please ensure images are in folders named 'real' or 'fake'.")
        os.remove(split_file)
        return

    # Load Model
    print(f"Loading model: {args.model}")
    model = DeepfakeEfficientNet(pretrained=False).to(DEVICE)
    checkpoint = torch.load(args.model, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        threshold = checkpoint.get("threshold", 0.5)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5
    model.eval()

    # Create Dataset and DataLoader
    dataset = DeepfakeDataset(split_file, transform=get_transforms(is_train=False))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    # Calculate metrics
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    eer, calc_threshold = calculate_eer(all_labels, all_preds)
    
    # Use threshold from model if available, else 0.5
    binary_preds = [1 if p > threshold else 0 for p in all_preds]
    acc = accuracy_score(all_labels, binary_preds)

    print("\n" + "="*50)
    print(f"CROSS-DATASET EVALUATION REPORT: {os.path.basename(args.dataset_dir)}")
    print("="*50)
    print(f"Total Samples: {len(all_labels)}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"AUC:           {roc_auc:.4f}")
    print(f"EER:           {eer:.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(all_labels, binary_preds, target_names=['Real', 'Fake']))

    # Clean up
    os.remove(split_file)

if __name__ == "__main__":
    main()
