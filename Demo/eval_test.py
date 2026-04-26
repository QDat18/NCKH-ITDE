import torch
import time
from torch.utils.data import DataLoader
from dataset_pytorch import DeepfakeDataset, get_transforms
from model_pytorch import DeepfakeEfficientNet

def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/best_pytorch_model_tuned.pth"
    BATCH_SIZE = 16

    test_ds = DeepfakeDataset("splits/test.txt", transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepfakeEfficientNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"Starting evaluation on {len(test_ds)} samples...")
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            _ = model(images)
    
    end_time = time.time()
    total_time = end_time - start_time
    num_batches = len(test_loader)
    avg_per_step = (total_time / num_batches) * 1000

    print("-" * 30)
    print(f"RESULTS FOR REPORT:")
    print(f"Total Batches: {num_batches}")
    print(f"Total Time:    {total_time:.2f} seconds")
    print(f"Avg Speed:     {avg_per_step:.2f} ms/step")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
