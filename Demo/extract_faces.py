import cv2
import os
from tqdm import tqdm

# ==============================
# 1. PATH CONFIG (LOCAL)
# ==============================
DATA_ROOT = "data_raw"
REAL_CELEB_SRC = os.path.join(DATA_ROOT, "Celeb-real")
REAL_YT_SRC    = os.path.join(DATA_ROOT, "YouTube-real")
FAKE_SRC       = os.path.join(DATA_ROOT, "Celeb-synthesis")

OUTPUT_ROOT = "data_image_train"
TARGET_SIZE = (380, 380)

# ==============================
# LOAD HAAR CASCADE
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def is_blurry(image, threshold=60):
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def extract_faces():
    configs = [
        {
            "label": "Real",
            "sources": [
                (REAL_CELEB_SRC, 35),
                (REAL_YT_SRC, 6),
                ("data_raw/Self-real", 20)
            ]
        },
        {
            "label": "Fake",
            "sources": [
                (FAKE_SRC, 4)
            ]
        }
    ]

    for cfg in configs:
        label = cfg["label"]
        out_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)

        for src_path, frames_per_vid in cfg["sources"]:
            if not os.path.exists(src_path):
                print(f"❌ Not found: {src_path}")
                continue

            videos = [v for v in os.listdir(src_path) if v.lower().endswith((".mp4", ".avi", ".mov"))]
            print(f"\n[INFO] {label} | {os.path.basename(src_path)} | {len(videos)} videos")

            for video in tqdm(videos):
                video_path = os.path.join(src_path, video)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total_frames // (frames_per_vid + 1))
                saved = 0
                
                for i in range(frames_per_vid):
                    frame_id = i * step
                    if frame_id >= total_frames:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        if len(faces) == 0:
                            continue
                            
                        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
                        
                        pad = int(0.20 * w)
                        x1, y1 = max(0, x - pad), max(0, y - pad)
                        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                        
                        crop = frame[y1:y2, x1:x2]
                        if is_blurry(crop):
                            continue
                        
                        crop = cv2.resize(crop, TARGET_SIZE)
                        out_name = f"{os.path.basename(src_path)}_{video}_f{saved}.jpg"
                        cv2.imwrite(os.path.join(out_dir, out_name), crop)
                        saved += 1
                    except Exception:
                        continue
                cap.release()

    print("\n[DONE] EXTRACTION COMPLETE (Haar Cascade)")

if __name__ == "__main__":
    extract_faces()
