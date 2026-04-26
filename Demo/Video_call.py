import cv2
import numpy as np
import os
import time
from collections import deque

# ================== CẤU HÌNH ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
import mediapipe as mp

MODEL_PATH = "models/best_model.h5"
V_WIDTH, V_HEIGHT = 640, 480
FRAME_SKIP = 3
TEMPORAL_WINDOW = 10
THRESHOLD = 0.5

# ================== LOAD MODEL ==================
print(f"⏳ Load model Keras {keras.__version__}...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model OK")

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
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    return float(model.predict(face, verbose=0)[0][0])

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
frame_id = 0
latency = 0

print("🚀 Running... Press Q to exit")

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

    cv2.imshow("Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
