import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import numpy as np
import os
import csv
from pathlib import Path
from collections import Counter, defaultdict

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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def make_frequency_tensor(image_tensor, mode="laplacian"):
    """
    Builds a 1-channel artifact map from the normalized RGB tensor.

    The default `laplacian` mode is a lightweight high-pass map. It keeps the
    branch cheap enough for local training while still exposing edge/frequency
    artifacts that RGB backbones may smooth over.
    """
    if mode in (None, "none"):
        return None
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Expected normalized RGB tensor [3,H,W], got {tuple(image_tensor.shape)}")

    image = image_tensor.detach().float().cpu()
    rgb = torch.clamp(image * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    gray = 0.2989 * rgb[0:1] + 0.5870 * rgb[1:2] + 0.1140 * rgb[2:3]

    if mode == "laplacian":
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        high_pass = F.conv2d(gray.unsqueeze(0), kernel, padding=1).squeeze(0).abs()
    elif mode == "fft":
        freq = torch.fft.fftshift(torch.fft.fft2(gray.squeeze(0)))
        high_pass = torch.log1p(torch.abs(freq)).unsqueeze(0)
        h, w = high_pass.shape[-2:]
        band_h = max(1, int(h * 0.08))
        band_w = max(1, int(w * 0.08))
        cy, cx = h // 2, w // 2
        high_pass[:, cy - band_h : cy + band_h + 1, cx - band_w : cx + band_w + 1] = 0.0
    else:
        raise ValueError(f"Unknown frequency mode: {mode}")

    mean = high_pass.mean()
    std = high_pass.std().clamp_min(1e-6)
    return (high_pass - mean) / std


class DeepfakeDataset(Dataset):
    def __init__(self, split_file, transform=None, hard_samples_file=None, frequency_mode="none"):
        self.samples = []
        self.transform = transform
        self.split_file = str(split_file)
        self.base_dir = Path(__file__).resolve().parent
        self.frequency_mode = frequency_mode or "none"
        
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                if ',' in text:
                    path, label = text.rsplit(',', 1)
                else:
                    path, label = text.rsplit(None, 1)
                self.samples.append((path, int(label)))

        hard_map = load_hard_sample_weights(hard_samples_file, self.base_dir) if hard_samples_file else {}
        self.labels = [label for _, label in self.samples]
        self.sources = [source_name(path) for path, _ in self.samples]
        self.source_labels = [(source, label) for source, label in zip(self.sources, self.labels)]
        self.sample_weights = []
        self.hard_types = []
        for path, _ in self.samples:
            weight, hard_type = lookup_hard_sample(hard_map, path, self.base_dir)
            self.sample_weights.append(weight)
            self.hard_types.append(hard_type)
        
        print(f"Loaded {len(self.samples)} samples from {split_file}")
        if hard_samples_file:
            hard_count = sum(1 for weight in self.sample_weights if weight > 1.0)
            print(f"Loaded {hard_count} hard-weighted samples from {hard_samples_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(self.resolve_sample_path(path)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            if self.frequency_mode != "none":
                freq = make_frequency_tensor(img, mode=self.frequency_mode)
                return (img, freq), label
            return img, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a zero tensor if image fails to load
            if self.frequency_mode != "none":
                return (torch.zeros((3, 380, 380)), torch.zeros((1, 380, 380))), label
            return torch.zeros((3, 380, 380)), label

    def resolve_sample_path(self, path):
        path_obj = Path(path)
        if path_obj.is_absolute() or path_obj.exists():
            return path_obj
        demo_path = self.base_dir / path_obj
        return demo_path if demo_path.exists() else path_obj

    def describe_distribution(self):
        return {
            "samples": len(self.samples),
            "labels": dict(sorted(Counter(self.labels).items())),
            "sources": dict(sorted(Counter(self.sources).items())),
            "source_labels": {
                f"{source}|{label}": count
                for (source, label), count in sorted(Counter(self.source_labels).items())
            },
            "hard_types": dict(sorted(Counter(self.hard_types).items())),
        }


def normalize_key(path, base_dir):
    raw = str(path).strip()
    keys = {raw.replace("/", "\\")}
    path_obj = Path(raw)
    if not path_obj.is_absolute():
        path_obj = Path(base_dir) / raw
    try:
        keys.add(str(path_obj.resolve()).replace("/", "\\").lower())
    except OSError:
        keys.add(str(path_obj.absolute()).replace("/", "\\").lower())
    return keys


def load_hard_sample_weights(hard_samples_file, base_dir):
    path = Path(hard_samples_file)
    if not path.is_absolute():
        path = Path(base_dir) / path
    if not path.exists():
        raise FileNotFoundError(f"Hard sample file not found: {hard_samples_file}")

    hard_map = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "hard_weight"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            raw_path = row["path"].strip()
            if not raw_path:
                continue
            weight = float(row.get("hard_weight") or 1.0)
            hard_type = row.get("hard_type", "hard_sample") or "hard_sample"
            value = (max(1.0, weight), hard_type)
            for key in normalize_key(raw_path, base_dir):
                hard_map[key] = value
    return hard_map


def lookup_hard_sample(hard_map, path, base_dir):
    if not hard_map:
        return 1.0, "normal"
    for key in normalize_key(path, base_dir):
        if key in hard_map:
            return hard_map[key]
    return 1.0, "normal"


def source_name(path):
    text = str(path)
    lower = text.lower()
    base = os.path.basename(text)
    if "FF_" in base:
        return "FaceForensics++"
    if "Celeb-" in base:
        return "Celeb-DF"
    if "dfdc" in lower or "deepfake-detection-challenge" in lower:
        return "DFDC"
    if "real and fake face" in lower or "ciplab" in lower:
        return "CIPLAB/Kaggle"
    return "Other"


def build_balanced_sampler(dataset, mode="source_label", max_weight=None):
    """
    Creates a WeightedRandomSampler for imbalanced deepfake data.

    mode:
    - label: balance Real/Fake only.
    - source: balance FaceForensics++/Celeb-DF/DFDC only.
    - source_label: balance each (source, label) cell most aggressively.
    - combined: gentler balance using sqrt(label_weight * source_weight).
    """
    if mode in (None, "none"):
        return None
    if mode not in {"label", "source", "source_label", "combined"}:
        raise ValueError(f"Unknown sampler mode: {mode}")

    labels = list(dataset.labels)
    sources = list(dataset.sources)
    source_labels = list(dataset.source_labels)
    total = len(labels)
    if total == 0:
        raise ValueError("Cannot build sampler for an empty dataset.")

    label_counts = Counter(labels)
    source_counts = Counter(sources)
    source_label_counts = Counter(source_labels)
    num_labels = len(label_counts)
    num_sources = len(source_counts)
    num_source_label_cells = len(source_label_counts)

    hard_weights = list(getattr(dataset, "sample_weights", [1.0] * total))

    weights = []
    for idx, (source, label) in enumerate(source_labels):
        label_weight = total / (num_labels * label_counts[label])
        source_weight = total / (num_sources * source_counts[source])
        source_label_weight = total / (num_source_label_cells * source_label_counts[(source, label)])

        if mode == "label":
            weight = label_weight
        elif mode == "source":
            weight = source_weight
        elif mode == "source_label":
            weight = source_label_weight
        else:
            weight = float(np.sqrt(label_weight * source_weight))
        weight *= float(hard_weights[idx])
        if max_weight is not None:
            weight = min(float(max_weight), weight)
        weights.append(weight)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=total,
        replacement=True,
    )


def sampler_weight_summary(dataset, mode="source_label", max_weight=None):
    labels = list(dataset.labels)
    sources = list(dataset.sources)
    source_labels = list(dataset.source_labels)
    total = len(labels)
    label_counts = Counter(labels)
    source_counts = Counter(sources)
    source_label_counts = Counter(source_labels)
    num_labels = len(label_counts) or 1
    num_sources = len(source_counts) or 1
    num_source_label_cells = len(source_label_counts) or 1

    hard_by_cell = defaultdict(list)
    for source_label, weight in zip(source_labels, getattr(dataset, "sample_weights", [1.0] * total)):
        hard_by_cell[source_label].append(float(weight))

    rows = []
    for source, label in sorted(source_label_counts):
        label_weight = total / (num_labels * label_counts[label])
        source_weight = total / (num_sources * source_counts[source])
        source_label_weight = total / (num_source_label_cells * source_label_counts[(source, label)])
        combined_weight = float(np.sqrt(label_weight * source_weight))
        avg_hard_weight = float(np.mean(hard_by_cell[(source, label)])) if hard_by_cell[(source, label)] else 1.0
        selected = {
            "label": label_weight,
            "source": source_weight,
            "source_label": source_label_weight,
            "combined": combined_weight,
        }.get(mode, 1.0)
        selected *= avg_hard_weight
        if max_weight is not None:
            selected = min(float(max_weight), selected)
        rows.append(
            {
                "source": source,
                "label": label,
                "samples": source_label_counts[(source, label)],
                "label_weight": label_weight,
                "source_weight": source_weight,
                "source_label_weight": source_label_weight,
                "combined_weight": combined_weight,
                "avg_hard_weight": avg_hard_weight,
                "selected_weight": selected,
            }
        )
    return rows

def get_transforms(is_train=True):
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if is_train:
        transform = A.Compose([
            A.Resize(380, 380),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            
            # 1. Advanced Robustness Augmentations
            A.OneOf([
                A.ImageCompression(quality_range=(30, 70), p=1.0),             # JPEG simulation
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),                      # Blur injection
                A.MotionBlur(blur_limit=7, p=1.0),                             # Motion blur
            ], p=0.5),
            
            # 2. Advanced Noise Augmentations (Crucial for fixing AUC drop on noise)
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
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
