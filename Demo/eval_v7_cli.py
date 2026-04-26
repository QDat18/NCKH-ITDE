import cv2
import torch
import numpy as np
import os
from collections import deque
from scipy.spatial.distance import cosine
from PIL import Image
from torchvision import transforms

# Copy core functions from video_app.py
TARGET_SIZE = (380, 380)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

from model_pytorch import DeepfakeEfficientNet

class IdentityTracker:
    def __init__(self, window_size=15):
        self.centroid_buffer = deque(maxlen=window_size)
        self.embed_buffer = deque(maxlen=window_size)
        self.risk_buffer = deque(maxlen=window_size)
        self.lap_buffer = deque(maxlen=window_size)
        
    def update(self, center, embed, raw_risk=0.0, lap_val=0.0):
        self.centroid_buffer.append(center); self.embed_buffer.append(embed)
        self.risk_buffer.append(raw_risk)
        self.lap_buffer.append(lap_val)
        
    def get_metrics(self, current_lap=None):
        drift = cosine(self.embed_buffer[-1], self.embed_buffer[-2]) if len(self.embed_buffer)>1 else 0.0
        jitter = np.std(np.linalg.norm(np.diff(np.array(self.centroid_buffer), axis=0), axis=1)) if len(self.centroid_buffer)>1 else 0.0
        smoothed_risk = np.median(self.risk_buffer) if len(self.risk_buffer) > 0 else 0.0
        z_lap = 0.0
        if current_lap is not None and len(self.lap_buffer) > 3:
            mean_lap = np.mean(self.lap_buffer)
            std_lap = np.std(self.lap_buffer) + 1e-6
            z_lap = (current_lap - mean_lap) / std_lap
        return drift, jitter, smoothed_risk, z_lap

def analyze_forensics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    fft = np.percentile(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1), 99) / (np.mean(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)) + 1e-6)
    v_lap = 1 / (1 + np.exp(-0.4 * (lap - 8.0) * -1))
    v_fft = 1 / (1 + np.exp(-0.5 * (fft - 3.2) * 1))
    return (v_lap + v_fft) / 2.0, lap

def load_model():
    model = DeepfakeEfficientNet(model_name='efficientnet_b4', pretrained=False)
    ckpt = torch.load('models/best_pytorch_model_final.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    return model.to(DEVICE).eval(), 0.57

def predict_expert(face_img, model):
    if face_img is None or face_img.size == 0: return 0.0, None
    tf = transforms.Compose([transforms.Resize(TARGET_SIZE), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img = tf(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, embed = model(img, return_features=True)
        return torch.sigmoid(logits).item(), embed.cpu().numpy().flatten()

def evaluate_video(video_path, model, thresh=0.57):
    print(f"\n--- SCANNING: {os.path.basename(video_path)} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("LỖI: Không thể mở video!")
        return
        
    scores, drifts = [], []
    curr = 0; tr3 = IdentityTracker()
    
    while cap.isOpened():
        ret, f = cap.read()
        if not ret: break
        curr += 1
        if curr % 15 != 0: continue
        
        f_r, lap_val = analyze_forensics(cv2.resize(f, (640, 480)))
        faces = face_cascade.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 1.1, 6)
        a_s = 0.0
        if len(faces) > 0:
            b = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
            a_s, em = predict_expert(f[b[1]:b[1]+b[3], b[0]:b[0]+b[2]], model)
            unc = 0.0
            raw_r = min(max(a_s, f_r) + unc, 1.0)
            tr3.update((0,0), em, raw_r, lap_val)
            drifts.append(tr3.get_metrics()[0])
            scores.append(tr3.get_metrics()[2])
        else:
            scores.append(f_r)
            drifts.append(0.0)
            
    cap.release()
    
    if not scores:
        print("Video không có dữ liệu để phân tích.")
        return
        
    scores = np.array(scores)
    drifts = np.array(drifts)

    weights = np.ones_like(scores)
    weights[drifts > 0.85] = 0.3

    p95 = np.percentile(scores, 95) if len(scores) > 0 else 0
    mean = np.average(scores, weights=weights) if len(scores) > 0 else 0
    std = np.sqrt(np.average((scores - mean)**2, weights=weights)) if len(scores) > 0 else 0

    high_ratio = np.mean(scores > thresh) if len(scores) > 0 else 0
    spikes = np.sum(scores > 0.8) if len(scores) > 0 else 0
    d_p95 = np.percentile(drifts, 95) if len(drifts) > 0 else 0

    is_fake = False
    reason = []

    if p95 > 0.75 and high_ratio > 0.15:
        is_fake = True
        reason.append("Nhiều frame có rủi ro cao")
    if spikes > 3:
        is_fake = True
        reason.append("Xuất hiện spike bất thường")
    if d_p95 > 0.55:
        is_fake = True
        reason.append("Lệch định danh cao")
    if mean < 0.4 and std < 0.15 and d_p95 <= 0.55:
        is_fake = False
        reason = ["Stable behavior, normal drift"]

    if is_fake:
        print(f"🚨 FAKE DETECTED")
        print(f"   Reason: {', '.join(reason)}")
    else:
        print(f"✅ REAL VIDEO")
    print(f"   P95 Score: {p95:.4f} | Mean: {mean:.4f} | Std: {std:.4f} | Spikes: {spikes}")
    print(f"   Drift P95: {d_p95:.4f}")

if __name__ == '__main__':
    model, thresh = load_model()
    v1 = r"D:\KyV_HocVienNganHang\NCKH\Final\video demo\7752331785432.mp4"
    v2 = r"D:\KyV_HocVienNganHang\NCKH\Final\video demo\7756129436549.mp4"
    evaluate_video(v1, model, thresh)
    evaluate_video(v2, model, thresh)
