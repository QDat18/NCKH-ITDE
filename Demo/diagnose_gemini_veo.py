import cv2
import torch
import numpy as np
import os
import pandas as pd
from model_pytorch import DeepfakeEfficientNet
from torchvision import transforms
from PIL import Image

# Metadata Config
VIDEO_PATH = r"D:\HoangQuagDat\Deepfake-Detection-Model\Tạo_video_faceswap_người_nổi_tiếng.mp4"
MODEL_PATH = "models/best_pytorch_model_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "diagnosis_results_v51.csv"

def analyze_global_forensics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Laplacian
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_score = np.mean(sobel_mag)
    
    # 3. FFT Spike (p99/mean)
    fft_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(fft_shift) + 1)
    p99 = np.percentile(magnitude, 99)
    mean = np.mean(magnitude)
    spike_score = p99 / (mean + 1e-6)
    
    # Normalization (v5.1 logic)
    s_lap = np.clip(lap_var / 800.0, 0, 1)
    s_sob = np.clip(sobel_score / 40.0, 0, 1)
    s_fft = np.clip(1.0 - (spike_score - 1.5)/2.0, 0, 1)
    
    global_realness = (s_lap * 0.4 + s_sob * 0.3 + s_fft * 0.3)
    global_fakeness = 1.0 - global_realness
    
    return global_fakeness, lap_var, sobel_score, spike_score, p99, mean

def diagnose():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    # Load Model with Dictionary Support
    model = DeepfakeEfficientNet(pretrained=False).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        threshold = checkpoint.get("threshold", 0.5)
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.5
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    results = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"System Optimized EER Threshold: {threshold}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % 5 != 0: continue # Higher density sampling
        
        # 1. Face Analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        face_score = 0
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_crop = frame[y:y+h, x:x+w]
            face_tensor = transform(Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                face_score = torch.sigmoid(model(face_tensor)).item()
        
        # 2. Global Forensic Analysis
        g_fake, lap, sob, spike, p99, mean = analyze_global_forensics(frame)
        
        # 3. Decision Logic Consensus
        is_ai_v51 = (lap < 15 and spike > 2.4) or (sob < 11 and spike > 2.1)
        
        results.append({
            "frame": frame_idx,
            "face_score": face_score,
            "global_score": g_fake,
            "lap": lap,
            "sobel": sob,
            "spike": spike,
            "p99": p99,
            "mean": mean,
            "is_ai_flag": is_ai_v51
        })
        if frame_idx % 50 == 0: print(f"Audit Progress: {frame_idx}/{total_frames}...")

    cap.release()
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    
    print("\nAUDIT V5.1 COMPLETE")
    print("="*40)
    print(f"AVG FACE SCORE:   {df['face_score'].mean():.4f}")
    print(f"AVG GLOBAL SCORE: {df['global_score'].mean():.4f}")
    print(f"AVG SPIKE SCORE:  {df['spike'].mean():.4f} (Max: {df['spike'].max():.4f})")
    print(f"AVG LAPLACIAN:    {df['lap'].mean():.4f} (Min: {df['lap'].min():.4f})")
    print(f"AVG SOBEL:        {df['sobel'].mean():.4f} (Min: {df['sobel'].min():.4f})")
    print(f"AI FLAGS DETECTED: {df['is_ai_flag'].sum()} frames")
    print("="*40)

if __name__ == "__main__":
    diagnose()
