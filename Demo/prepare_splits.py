import os
import random
from pathlib import Path

# Paths
ROOT_DIR = Path("d:/KyV_HocVienNganHang/NCKH/Final/Demo")
# We ONLY use the extracted face crops for training
DATA_DIR = ROOT_DIR / "data_image_train"
SPLITS_DIR = ROOT_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

samples = []

# 1. Collect all extracted faces
all_real = []
all_fake = []

if DATA_DIR.exists():
    for label_name, label in [("Real", 0), ("Fake", 1)]:
        dir_path = DATA_DIR / label_name
        if dir_path.exists():
            files = [str(f.absolute()) for f in dir_path.glob("*.jpg")]
            if label == 0:
                all_real.extend([(f, label) for f in files])
            else:
                all_fake.extend([(f, label) for f in files])

print(f"Total Extracted Real Faces: {len(all_real)}")
print(f"Total Extracted Fake Faces: {len(all_fake)}")

# 2. Balancing Logic
random.seed(42)
# We want to use all available Real data and a balanced amount of Fake data
# Given the high false positive rate, keeping a 1:1 or 1:2 ratio is good.
# Let's keep it 1:1 for the most robust base model.

if len(all_fake) > len(all_real):
    print(f"Sampling Fake data to match {len(all_real)} real samples for perfect balance...")
    all_fake = random.sample(all_fake, len(all_real))
else:
    print(f"Sampling Real data to match {len(all_fake)} fake samples...")
    all_real = random.sample(all_real, len(all_fake))

samples = all_real + all_fake

# 3. Shuffle and Split (80/10/10)
random.seed(42)
random.shuffle(samples)

total = len(samples)
train_size = int(0.8 * total)
val_size = int(0.1 * total)

train_samples = samples[:train_size]
val_samples = samples[train_size:train_size + val_size]
test_samples = samples[train_size + val_size:]

def save_split(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for p, l in data:
            f.write(f"{p} {l}\n")

save_split(SPLITS_DIR / "train.txt", train_samples)
save_split(SPLITS_DIR / "val.txt", val_samples)
save_split(SPLITS_DIR / "test.txt", test_samples)

print(f"\nFinal Balanced Splits:")
print(f"  Train: {len(train_samples)}")
print(f"  Val:   {len(val_samples)}")
print(f"  Test:  {len(test_samples)}")
print(f"Total Images for Training: {total}")
