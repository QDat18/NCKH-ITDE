import cv2
import numpy as np
import torch
import time
import pyvirtualcam
from collections import deque
from model_pytorch import DeepfakeEfficientNet
from torchvision import transforms
from PIL import Image

# mediapipe removed due to compatibility issues with Python 3.13

# ================== CONFIGURATION ==================
MODEL_PATH = "models/best_pytorch_model_v2.pth"
V_WIDTH, V_HEIGHT = 640, 480
FRAME_SKIP = 3
TEMPORAL_WINDOW = 10
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== LOAD MODEL ==================
print(f"Loading PyTorch model on {DEVICE}...")
model = DeepfakeEfficientNet(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model OK")

# ================== PREPROCESSING ==================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================== FACE DETECTION (OpenCV) ==================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not find haarcascade file. Check OpenCV installation.")
else:
    print("OpenCV FaceDetection OK")

label_queue = deque(maxlen=TEMPORAL_WINDOW)

def get_face_crop(frame, box, padding=0.15):
    x, y, w, h = box
    pad = int(padding * w)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    return frame[y1:y2, x1:x2]

# ================== PREDICTION ==================
def predict_deepfake(face):
    # Dùng face gốc (BGR) để Image.fromarray tự hiểu là RGB (giống lúc train lưu imwrite rồi load PIL)
    pil_img = Image.fromarray(face)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    return prob

# ================== MAIN LOOP ==================
print("Starting Virtual Camera... Use this camera in Google Meet/Zoom")
print("Press Ctrl+C in this terminal to stop.")

cap = cv2.VideoCapture(0)
frame_id = 0
latency = 0

try:
    with pyvirtualcam.Camera(width=V_WIDTH, height=V_HEIGHT, fps=20) as cam:
        print(f'Virtual Camera active: {cam.device}')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (V_WIDTH, V_HEIGHT))
            
            # Face Detection (OpenCV) - Tăng cường để tránh bắt nhầm vùng nhỏ
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=6, 
                minSize=(120, 120)
            )
            
            text = "Scanning..."
            color = (200, 200, 200)

            if len(faces) > 0:
                # Ưu tiên khuôn mặt lớn nhất
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                for (x, y, w, h) in faces[:1]: # Chỉ xử lý mặt chính
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(V_WIDTH, x + w), min(V_HEIGHT, y + h)
                    face = frame[y1:y2, x1:x2]

                    if face.size > 0:
                        if frame_id % FRAME_SKIP == 0:
                            t0 = time.time()
                            face_crop = get_face_crop(frame, (x, y, w, h))
                            score = predict_deepfake(face_crop)
                            latency = (time.time() - t0) * 1000
                            label_queue.append(score)

                        if label_queue:
                            avg = np.mean(label_queue)
                            if avg > THRESHOLD:
                                text = f"FAKE ({avg:.2f})"
                                color = (0, 0, 255) # Red
                            else:
                                text = f"REAL ({1-avg:.2f})"
                                color = (0, 255, 0) # Green

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                label_queue.clear()

            # Show latency
            cv2.putText(frame, f"Latency: {latency:.1f}ms",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

            # Convert BGR (OpenCV) to RGB (Virtual Cam)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Send frame to Virtual Camera
            cam.send(frame_rgb)
            cam.sleep_until_next_frame()

            frame_id += 1
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    print("Virtual Camera stopped.")
