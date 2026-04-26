import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms
from model_pytorch import DeepfakeEfficientNet
from tqdm import tqdm
import random

# ==============================
# CONFIG
# ==============================
BASE_MODEL_PATH = "models/best_pytorch_model.pth"
TUNED_MODEL_PATH = "models/best_pytorch_model_tuned.pth"
SELF_REAL_DIR = "data_image_train/Real" # After extraction, self-real will be here
# Note: In a real scenario, we might want to isolate self-real 
# but for now we assume the Real folder contains the self-recorded faces.

BATCH_SIZE = 4  # Keep it small for 6GB VRAM
EPOCHS = 5      # Short tuning for personalization
LR = 5e-6       # Extremely low learning rate to preserve FF++ knowledge
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (380, 380)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return torch.mean(self.alpha * (1 - pt)**self.gamma * BCE_loss)

# ==============================
# DATASET
# ==============================
class SelfTuningDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        
        # 1. Identify files that came from Self-real
        all_real = [os.path.join(SELF_REAL_DIR, f) for f in os.listdir(SELF_REAL_DIR) if "Self-real" in f]
        if not all_real:
            # Fallback to all real images if no specific Self-real tag found
            all_real = [os.path.join(SELF_REAL_DIR, f) for f in os.listdir(SELF_REAL_DIR)]
            
        for f in all_real:
            self.samples.append((f, 0)) # 0 = Real
            
        print(f"Added {len(all_real)} Self-Real samples for tuning.")
        
        # 2. Add an equal number of Fake samples from the training set to prevent bias
        fake_dir = "data_image_train/Fake"
        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
            random_fake = random.sample(fake_files, min(len(fake_files), len(all_real)))
            for f in random_fake:
                self.samples.append((f, 1)) # 1 = Fake
            print(f"Added {len(random_fake)} Fake samples for balance.")
            
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            # Handle potential broken images
            return torch.zeros((3, TARGET_SIZE[0], TARGET_SIZE[1])), label

def tuning():
    # 1. Transforms (Match training but stricter)
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SelfTuningDataset(transform=transform)
    if len(dataset) == 0:
        print("Error: No data found for tuning.")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Load Base B4 Model
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = DeepfakeEfficientNet(pretrained=False) # B4 is defined in model_pytorch.py
    try:
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Could not load model: {e}")
        return
        
    model.to(DEVICE)
    model.train()

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Tuning Loop
    print(f"Starting Personalization Tuning for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(loss=loss.item(), acc=correct/total)

    # 4. Save
    torch.save(model.state_dict(), TUNED_MODEL_PATH)
    print(f"✅ Personalization complete! Tuned model saved to: {TUNED_MODEL_PATH}")

if __name__ == "__main__":
    tuning()
