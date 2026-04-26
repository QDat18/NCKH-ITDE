import cv2
import os
import numpy as np
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
# Thư mục chứa 4 video của bạn
INPUT_DIR = r"D:\HoangQuagDat\Deepfake-Detection-Model"
OUTPUT_DIR = "data_finetune/Fake"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Lấy danh sách tất cả file .mp4
    video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    print(f"Found {len(video_files)} videos in {INPUT_DIR}")

    total_saved = 0
    
    for video_name in video_files:
        video_path = os.path.join(INPUT_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n--- Processing: {video_name} ({total_frames} frames) ---")
        
        saved_count = 0
        frame_idx = 0
        
        # Với NCKH, ta trích xuất khoảng 30-50 mặt mỗi video là đủ đa dạng
        max_faces_per_vid = 50 
        step = max(1, total_frames // max_faces_per_vid)

        while cap.isOpened() and saved_count < max_faces_per_vid:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for i, (x, y, w, h) in enumerate(faces):
                    pad = int(0.15 * w)
                    x1, y1 = max(0, x-pad), max(0, y-pad)
                    x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop = cv2.resize(crop, (224, 224))
                        # Lưu tên file bao gồm cả tên video để không bị trùng
                        out_name = f"gemini_{video_name}_f{frame_idx}_{i}.jpg"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), crop)
                        saved_count += 1
                        total_saved += 1
                        if saved_count >= max_faces_per_vid: break

            frame_idx += 1
            
        cap.release()
        print(f"Done: {video_name} -> Extracted {saved_count} faces")

    print(f"\n✅ FINISHED! Total {total_saved} faces extracted into {OUTPUT_DIR}")

if __name__ == "__main__":
    extract()


