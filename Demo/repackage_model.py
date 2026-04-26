import torch
import os
import pandas as pd

def repackage():
    model_path = "models/best_pytorch_model_final.pth"
    log_path = "training_log_clean.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(log_path):
        print("Error: Model or Log not found.")
        return

    # 1. Find best threshold from log
    df = pd.read_csv(log_path)
    # Get row with minimum val_eer
    best_row = df.loc[df['val_eer'].idxmin()]
    best_threshold = float(best_row['threshold'])
    best_eer = float(best_row['val_eer'])
    
    print(f"Detected Best EER: {best_eer:.4f} at Threshold: {best_threshold:.4f}")

    # 2. Load current state_dict
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # 3. Create new package
    data = {
        "model": state_dict,
        "threshold": best_threshold,
        "eer": best_eer
    }
    
    # 4. Save back
    torch.save(data, model_path)
    print(f"Successfully repackaged {model_path} with EER threshold.")

if __name__ == "__main__":
    repackage()
