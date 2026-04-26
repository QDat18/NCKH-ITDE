import cv2
import os
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DATA_ROOT = "data_raw/Self-real"
OUTPUT_ROOT = "data_image_train/Real"
TARGET_SIZE = (380, 380)
FRAMES_PER_VID = 30 # Take more frames for better personalization

# Initialize Haar Cascade (using Haar for reliability as established earlier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def is_blurry(image, threshold=60):
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def extract_self():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Not found: {DATA_ROOT}")
        return

    videos = [v for v in os.listdir(DATA_ROOT) if v.lower().endswith((".mp4", ".avi", ".mov"))]
    print(f"\n[INFO] Extracting Personal Real Data | {len(videos)} videos")

    for video in tqdm(videos):
        video_path = os.path.join(DATA_ROOT, video)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // (FRAMES_PER_VID + 1))
        saved = 0
        
        for i in range(FRAMES_PER_VID):
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
                
                # IMPORTANT: Keep "Self-real" in the filename so finetune_self.py can find it
                out_name = f"Self-real_{video}_f{saved}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_ROOT, out_name), crop)
                saved += 1
            except Exception:
                continue
        cap.release()

    print("\n✅ PERSONAL DATA EXTRACTION COMPLETE")

if __name__ == "__main__":
    extract_self()
