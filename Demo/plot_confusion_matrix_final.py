import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet
import os

def plot_final_confusion_matrix():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/best_pytorch_model_final.pth"
    TEST_FILE = "splits/test.txt"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = DeepfakeEfficientNet(pretrained=False).to(DEVICE)
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        threshold = checkpoint.get("threshold", 0.5)
        print(f"Using Optimal EER Threshold: {threshold:.4f}")
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5
        print("Warning: Using default 0.5 threshold")
    
    model.eval()
    
    # Load Test Data
    dataset = DeepfakeDataset(TEST_FILE, transform=get_transforms(is_train=False))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = probs > threshold 
            all_preds.extend(preds.astype(int))
            all_labels.extend(labels.numpy())

    # Compute CM
    cm = confusion_matrix(all_labels, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize

    # Plot
    plt.figure(figsize=(10, 8))
    labels_text = [f"{v}\n({p:.2%})" for v, p in zip(cm.flatten(), cm_percent.flatten())]
    labels_text = np.array(labels_text).reshape(2, 2)
    
    sns.heatmap(cm, annot=labels_text, fmt="", cmap='Blues', 
                xticklabels=['Thật (Real)', 'Giả (Fake)'], 
                yticklabels=['Thật (Real)', 'Giả (Fake)'])
    
    plt.title('MA TRẬN NHẦM LẪN - DEEPFAKE DETECTION FINAL', fontsize=15)
    plt.ylabel('Thực tế (Actual)', fontsize=12)
    plt.xlabel('Dự đoán (Predicted)', fontsize=12)
    
    save_path = "final_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success: Confusion Matrix saved to {save_path}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

if __name__ == "__main__":
    plot_final_confusion_matrix()
