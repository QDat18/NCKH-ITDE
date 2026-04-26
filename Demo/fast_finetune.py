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
CURRENT_MODEL_PATH = "models/best_pytorch_model.pth"
NEW_MODEL_PATH = "models/best_pytorch_model_v2.pth"
FINETUNE_DATA_DIR = "data_finetune/Fake"
REAL_DATA_DIR = "data_image_train/Real"

BATCH_SIZE = 8  # Keep it small for stability
EPOCHS = 10
LR = 1e-5       # Very low learning rate for fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# CUSTOM DATASET
# ==============================
class FinetuneDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        
        # 1. Add ALL new Gemini faces
        gemini_files = [os.path.join(FINETUNE_DATA_DIR, f) for f in os.listdir(FINETUNE_DATA_DIR)]
        for f in gemini_files:
            self.samples.append((f, 1)) # 1 = Fake
            
        print(f"Added {len(gemini_files)} new Fake samples.")
        
        # 2. Add an EQUAL number of random Real samples to maintain balance
        real_files = [os.path.join(REAL_DATA_DIR, f) for f in os.listdir(REAL_DATA_DIR)]
        random_real = random.sample(real_files, min(len(real_files), len(gemini_files)))
        for f in random_real:
            self.samples.append((f, 0)) # 0 = Real
            
        print(f"Added {len(random_real)} Real samples for balance.")
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def finetune():
    # 1. Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FinetuneDataset(transform=transform)
    if len(dataset) == 0:
        print("Error: No data found for fine-tuning.")
        return
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Load Model
    print(f"Loading base model from {CURRENT_MODEL_PATH}...")
    model = DeepfakeEfficientNet(pretrained=False)
    model.load_state_dict(torch.load(CURRENT_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Training Loop
    print(f"Starting Fine-tuning for {EPOCHS} epochs...")
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
    torch.save(model.state_dict(), NEW_MODEL_PATH)
    print(f"✅ Fine-tuning complete! Model saved to: {NEW_MODEL_PATH}")
    print("TIP: To use this model in the App, rename it or update MODEL_PATH in video_app.py")

if __name__ == "__main__":
    finetune()
