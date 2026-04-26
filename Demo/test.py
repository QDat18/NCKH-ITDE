import cv2
import numpy as np
import keras
import os

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/best_model.h5"
IMG_REAL = "data_image_train/Real/hoasonquy.jpg"   # ảnh mặt thật
IMG_FAKE = "data_image_train/Fake/Celeb-synthesis_id0_id1_0002.mp4_f0.jpg"   # ảnh deepfake
IMG_SIZE = 224

# ==============================
# LOAD MODEL
# ==============================
print("⏳ Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

# ==============================
# PREPROCESS (PHẢI GIỐNG TRAIN)
# ==============================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# TEST FUNCTION
# ==============================
def test_image(path, label_name):
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy {path}")
        return

    img = cv2.imread(path)
    x = preprocess(img)
    pred = model.predict(x, verbose=0)[0][0]

    print(f"\n🖼 {label_name}")
    print(f"→ Raw score: {pred:.4f}")

    if pred > 0.7:
        print("→ Predict: FAKE")
    elif pred < 0.4:
        print("→ Predict: REAL")
    else:
        print("→ Predict: UNCERTAIN")

# ==============================
# RUN TEST
# ==============================
print("\n====== TEST MODEL ======")
test_image(IMG_REAL, "REAL IMAGE")
test_image(IMG_FAKE, "FAKE IMAGE")
print("\n====== DONE ======")
