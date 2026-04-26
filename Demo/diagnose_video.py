import cv2
import torch
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
from torchvision import transforms
from model_pytorch import DeepfakeEfficientNet
import time

# Sửa lỗi hiển thị tiếng Việt trên Terminal Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ================== CONFIGURATION ==================
MODEL_PATH = "models/best_pytorch_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV = "diagnosis_results.csv"
BEST_FAKE_IMG = "highest_fake_score_frame.jpg"

# Preprocessing (Consistent with training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def diagnose(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    # 1. Load Model
    print(f"Loading model on {DEVICE}...")
    model = DeepfakeEfficientNet(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. Initialize Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Read Video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Starting diagnosis for: {video_path}")
    print(f"Total Frames: {total_frames} | FPS: {fps:.1f}")

    results = []
    max_score = -1.0
    best_frame = None
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every 3rd frame to speed up, or every frame for precision
        # For diagnosis, let's do every 5th frame to be fast but representative
        if frame_idx % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Take the largest face
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                (x, y, w, h) = faces[0]
                
                face_img = frame[y:y+h, x:x+w]
                if face_img.size > 0:
                    # Preprocess
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
                    
                    # Predict
                    with torch.no_grad():
                        output = model(input_tensor)
                        prob = torch.sigmoid(output).item()
                    
                    results.append({
                        "frame": frame_idx,
                        "timestamp_sec": frame_idx / fps,
                        "fake_probability": prob
                    })
                    
                    if prob > max_score:
                        max_score = prob
                        best_frame = frame.copy()
                        # Draw box on the best frame for visualization
                        cv2.rectangle(best_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        cv2.putText(best_frame, f"Score: {prob:.4f}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    # 4. Save & Report
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n--- DIAGNOSIS COMPLETE ---")
        print(f"Results saved to: {OUTPUT_CSV}")
        
        avg_score = df["fake_probability"].mean()
        max_score_val = df["fake_probability"].max()
        
        print(f"Average Fake Probability: {avg_score:.4f}")
        print(f"Maximum Fake Probability: {max_score_val:.4f}")
        
        if best_frame is not None:
            cv2.imwrite(BEST_FAKE_IMG, best_frame)
            print(f"Saved highest-score frame to: {BEST_FAKE_IMG}")
            
        print("\nInterpretation:")
        if max_score_val < 0.2:
            print("(!) Model is EXTREMELY confident this is REAL (Scores < 0.2).")
            print("Possible reason: The AI artifacts in Gemini videos are too subtle for the current backbone.")
        elif max_score_val < 0.5:
            print("(!) Model is leaning towards REAL but sees some suspicious features (Scores 0.2 - 0.5).")
        else:
            print("OK. Model detected the fake (Scores > 0.5).")
    else:
        print("No faces detected in the video.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_video.py <video_path>")
    else:
        diagnose(sys.argv[1])
