import cv2
import numpy as np
import torch
import time
from collections import deque
import mediapipe as mp
from model_pytorch import DeepfakeEfficientNet
from torchvision import transforms
from PIL import Image

# ================== CẤU HÌNH ==================
MODEL_PATH = "models/best_pytorch_model_tuned.pth"
V_WIDTH, V_HEIGHT = 640, 480
FRAME_SKIP = 3
TEMPORAL_WINDOW = 10
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== LOAD MODEL ==================
print(f"⏳ Loading PyTorch model...")
model = DeepfakeEfficientNet(model_name='efficientnet_b4', pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model OK")

# ================== PREPROCESSING ==================
preprocess = transforms.Compose([
    transforms.Resize((380, 380)), # B4 Native resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================== MEDIAPIPE ==================
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)
print("✅ MediaPipe FaceDetection OK")

label_queue = deque(maxlen=TEMPORAL_WINDOW)

# ================== DỰ ĐOÁN ==================
def predict_deepfake(face):
    # Convert BGR (OpenCV) to RGB then to PIL
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    return prob

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
frame_id = 0
latency = 0

print("🚀 Running PyTorch Inference... Press Q to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (V_WIDTH, V_HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)
    text = "Scanning..."
    color = (200, 200, 200)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x, y, w, h = int(bbox.xmin * V_WIDTH), int(bbox.ymin * V_HEIGHT), \
                         int(bbox.width * V_WIDTH), int(bbox.height * V_HEIGHT)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(V_WIDTH, x + w), min(V_HEIGHT, y + h)
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                if frame_id % FRAME_SKIP == 0:
                    t0 = time.time()
                    score = predict_deepfake(face)
                    latency = (time.time() - t0) * 1000
                    label_queue.append(score)

                if label_queue:
                    avg = np.mean(label_queue)
                    if avg > THRESHOLD:
                        text = f"FAKE ({avg:.2f})"
                        color = (0, 0, 255)
                    else:
                        text = f"REAL ({1-avg:.2f})"
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        label_queue.clear()

    cv2.putText(frame, f"Latency: {latency:.1f}ms",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)

    cv2.imshow("Deepfake Detection (PyTorch)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
