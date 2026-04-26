import torch
from torch.utils.data import DataLoader
from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet
from sklearn.metrics import confusion_matrix
import os

def get_counts():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/best_pytorch_model.pth"
    BATCH_SIZE = 16

    test_ds = DeepfakeDataset("splits/test.txt", transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepfakeEfficientNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            binary_preds = [1 if p > 0.5 else 0 for p in probs]
            all_preds.extend(binary_preds)
            all_labels.extend(labels.numpy())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    print("-" * 30)
    print(f"CONFUSION MATRIX COUNTS:")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print("-" * 30)

if __name__ == "__main__":
    get_counts()
