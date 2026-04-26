import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyvirtualcam
from collections import defaultdict, deque

# ======================================================
# 1. CẤU HÌNH HỆ THỐNG
# ======================================================
# MODEL_PATH = 'models/nckh_model.keras'
MODEL_PATH = 'models/best_model.h5'
V_WIDTH, V_HEIGHT = 640, 480
FPS = 24
THRESHOLD = 0.5
BUFFER_SIZE = 10

# ======================================================
# 2. KHỞI TẠO MEDIAPIPE & MODEL
# ======================================================
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

print("⏳ Đang nạp mô hình AI...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

# ======================================================
# 3. BUFFER RIÊNG CHO TỪNG KHUÔN MẶT
# ======================================================
face_buffers = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))

def get_face_id(x, y, w, h):
    """
    Gán ID tạm thời cho khuôn mặt dựa trên vị trí
    Đủ ổn định cho video call
    """
    cx = x + w // 2
    cy = y + h // 2
    return f"{cx//50}_{cy//50}"

# ======================================================
# 4. KHỞI TẠO WEBCAM & VIRTUAL CAMERA
# ======================================================
cap = cv2.VideoCapture(0)

with pyvirtualcam.Camera(width=V_WIDTH, height=V_HEIGHT, fps=FPS) as vcam:
    print(f"🎥 Virtual Camera: {vcam.device}")
    print("👉 Mở Google Meet / Zoom → chọn OBS Virtual Camera")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (V_WIDTH, V_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # 5. PHÁT HIỆN KHUÔN MẶT
        # --------------------------------------------------
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * V_WIDTH)
                y = int(bbox.ymin * V_HEIGHT)
                w = int(bbox.width * V_WIDTH)
                h = int(bbox.height * V_HEIGHT)

                x, y = max(0, x), max(0, y)

                face = frame[y:y+h, x:x+w]

                if face.size == 0:
                    continue

                # --------------------------------------------------
                # 6. TIỀN XỬ LÝ & DỰ ĐOÁN
                # --------------------------------------------------
                face_input = cv2.resize(face, (224, 224))
                face_input = face_input / 255.0
                face_input = np.expand_dims(face_input, axis=0)

                prob_fake = model.predict(face_input, verbose=0)[0][0]

                # --------------------------------------------------
                # 7. TEMPORAL VOTING RIÊNG
                # --------------------------------------------------
                face_id = get_face_id(x, y, w, h)
                face_buffers[face_id].append(prob_fake)
                avg_prob = np.mean(face_buffers[face_id])

                # --------------------------------------------------
                # 8. GÁN NHÃN
                # --------------------------------------------------
                if avg_prob > THRESHOLD:
                    label = f"FAKE ({avg_prob:.2f})"
                    color = (0, 0, 255)
                else:
                    label = f"REAL ({1 - avg_prob:.2f})"
                    color = (0, 255, 0)

                # --------------------------------------------------
                # 9. HIỂN THỊ
                # --------------------------------------------------
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # --------------------------------------------------
        # 10. GỬI FRAME SANG VIRTUAL CAMERA
        # --------------------------------------------------
        vcam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vcam.sleep_until_next_frame()

cap.release()
print("🛑 Đã dừng hệ thống")
