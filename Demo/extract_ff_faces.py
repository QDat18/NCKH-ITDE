import cv2
import os
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
FF_ROOT = r"D:\KyV_HocVienNganHang\NCKH\Project_part2\frames"
OUTPUT_ROOT = "data_image_train"
TARGET_SIZE = (380, 380)
FRAMES_PER_FOLDER = 5

# ==============================
# LOAD HAAR CASCADE
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def is_blurry(image, threshold=80):
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def extract_ff_faces():
    categories = ["fake", "real"]
    
    for cat in categories:
        src_cat_path = os.path.join(FF_ROOT, cat)
        label = "Fake" if cat == "fake" else "Real"
        out_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        
        if not os.path.exists(src_cat_path):
            print(f"❌ Not found: {src_cat_path}")
            continue
            
        folders = os.listdir(src_cat_path)
        print(f"\n[INFO] Processing FF++ {label} | {len(folders)} folders (Haar Cascade)")
        
        for folder in tqdm(folders):
            folder_path = os.path.join(src_cat_path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not images:
                continue
                
            step = max(1, len(images) // (FRAMES_PER_FOLDER + 1))
            saved = 0
            
            for i in range(FRAMES_PER_FOLDER):
                idx = i * step
                if idx >= len(images):
                    break
                    
                img_name = images[idx]
                img_path = os.path.join(folder_path, img_name)
                
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) == 0:
                        continue
                    
                    # Take the largest face
                    (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
                    
                    # Padding (20%)
                    pad = int(0.20 * w)
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                    
                    crop = frame[y1:y2, x1:x2]
                    if is_blurry(crop, threshold=60): # Lower threshold for Haar crops
                        continue
                        
                    crop = cv2.resize(crop, TARGET_SIZE)
                    out_name = f"FF_{cat}_{folder}_f{saved}.jpg"
                    cv2.imwrite(os.path.join(out_dir, out_name), crop)
                    saved += 1
                except Exception:
                    continue

    print("\n[DONE] FF++ EXTRACTION COMPLETE (Haar Cascade)")

if __name__ == "__main__":
    extract_ff_faces()
