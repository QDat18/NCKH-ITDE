import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from dataset_pytorch import get_transforms
from model_pytorch import DeepfakeEfficientNet
from sklearn.metrics import roc_curve, auc
import os
import io

class RobustnessDataset(Dataset):
    def __init__(self, split_file, transform=None, degradation=None):
        self.samples = []
        self.transform = transform
        self.degradation = degradation
        
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def apply_degradation(self, img):
        if self.degradation == "compression":
            # Simulate JPEG Compression (TikTok/YouTube)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=30)
            buffer.seek(0)
            return Image.open(buffer)
        
        elif self.degradation == "blur":
            # Simulate out-of-focus webcam
            return img.filter(ImageFilter.GaussianBlur(radius=2))
        
        elif self.degradation == "noise":
            # Simulate low-light sensor noise (Realistic std=8 instead of extreme std=25)
            img_np = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 8, img_np.shape)
            noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
            
        elif self.degradation == "dark":
            # Simulate low light Zoom call
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(0.5)
            
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        
        if self.degradation:
            img = self.apply_degradation(img)
            
        if self.transform:
            img = self.transform(img)
        return img, label

def evaluate_robustness():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/best_pytorch_model_final.pth"
    TEST_FILE = "splits/test.txt"
    
    model = DeepfakeEfficientNet(pretrained=False).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    transform = get_transforms(is_train=False)
    
    tests = [
        ("Original", None),
        ("High Compression (Q=30)", "compression"),
        ("Gaussian Blur (R=2)", "blur"),
        ("Sensor Noise (Gauss)", "noise"),
        ("Low Light (-50%)", "dark")
    ]
    
    print("\n" + "="*60)
    print("MODEL ROBUSTNESS STRESS TEST REPORT")
    print("="*60)
    print(f"{'Condition':<25} | {'Samples':<8} | {'AUC':<8}")
    print("-" * 60)
    
    for name, deg in tests:
        ds = RobustnessDataset(TEST_FILE, transform=transform, degradation=deg)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dl:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.numpy())
        
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        print(f"{name:<25} | {len(all_labels):<8} | {roc_auc:.4f}")

    print("="*60)

if __name__ == "__main__":
    evaluate_robustness()
