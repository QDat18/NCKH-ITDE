# !pip install mtcnn

import cv2 
import os 
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Cau hinh
INPUT_DIR = '/content/drive/MyDrive/Deepfake_Project/my_raw_photos'
OUTPUT_DIR = '/content/drive/MyDrive/Deepfake_Project/my_processed_faces'
IMG_SIZE = 224

detector = MTCNN()
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Đang xử lý ảnh từ thư mục: {INPUT_DIR}")

count = 0
for filename in os.listdir(INPUT_DIR): 
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Không thể đọc ảnh: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Phát hiện khuôn mặt
        results = detector.detect_faces(img_rgb)
        if results:
            # Lấy khuôn mặt đầu tiên được phát hiện
            x, y, w, h = results[0]['box']
            padding = int(w * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = w + 2 * padding
            h = h + 2 * padding

            face = img_rgb[y:y+h, x:x+w]
            if face.size == 0:
                print(f"Khuôn mặt không hợp lệ trong ảnh: {img_path}")
                continue

            try:
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                save_path = os.path.join(OUTPUT_DIR, f"me_{count}.jpg")
                cv2.imwrite(save_path, face_resized)
                count += 1
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_path}: {e}")
print(f"Đã xử lý xong {count} khuôn mặt và lưu vào thư mục: {OUTPUT_DIR}")