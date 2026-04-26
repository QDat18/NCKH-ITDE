import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsWrapper:
    """
    Wraps Albumentations transforms to be compatible with torchvision pipelines.
    Expects PIL Images, converts to NumPy, applies transform, and returns a PyTorch tensor.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        augmented = self.transform(image=img)
        return augmented['image']

class DeepfakeDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.samples = []
        self.transform = transform
        
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))
        
        print(f"Loaded {len(self.samples)} samples from {split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a zero tensor if image fails to load
            return torch.zeros((3, 380, 380)), label

def get_transforms(is_train=True):
    if is_train:
        transform = A.Compose([
            A.Resize(380, 380),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            
            # 1. Advanced Robustness Augmentations
            A.OneOf([
                A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0), # JPEG Simulation
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),                      # Blur injection
                A.MotionBlur(blur_limit=7, p=1.0),                             # Motion blur
            ], p=0.5),
            
            # 2. Advanced Noise Augmentations (Crucial for fixing AUC drop on noise)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.5),
            
            # 3. Low-light and Lighting Variations
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(380, 380),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return AlbumentationsWrapper(transform)
