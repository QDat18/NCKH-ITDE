import os
import tensorflow as tf

# Vẫn giữ Patch vì file .h5 của bạn có cấu trúc cũ
from tensorflow.keras.layers import InputLayer
original_init = InputLayer.__init__
def patched_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    original_init(self, *args, **kwargs)
InputLayer.__init__ = patched_init

KERAS_MODEL_PATH = "models/best_model.h5"
TEMP_KERAS_PATH = "models/temp_model.keras"
TFLITE_MODEL_PATH = "models/best_model_fp16.tflite"

print("⏳ Bước 1: Đang tải và chuẩn hóa mô hình...")
try:
    # Tải model .h5 cũ
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
    
    # Lưu tạm sang định dạng .keras (định dạng mới nhất, giúp fix lỗi cấu trúc)
    model.save(TEMP_KERAS_PATH)
    
    # Tải lại chính cái temp đó
    model = tf.keras.models.load_model(TEMP_KERAS_PATH)
    print("✅ Chuẩn hóa mô hình thành công!")
except Exception as e:
    print(f"❌ Lỗi bước 1: {e}")
    exit()

print("⚙️ Bước 2: Đang chuyển đổi sang TFLite...")
try:
    # Sử dụng bộ chuyển đổi từ model đã chuẩn hóa
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Tối ưu hóa
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    
    # Xóa file tạm cho sạch máy
    if os.path.exists(TEMP_KERAS_PATH):
        os.remove(TEMP_KERAS_PATH)
        
    print("-" * 30)
    print(f"✅ CHUYỂN ĐỔI THÀNH CÔNG!")
    print(f"📁 File: {TFLITE_MODEL_PATH}")
    print("-" * 30)

except Exception as e:
    print(f"❌ Lỗi bước 2: {e}")
    print("\n💡 Gợi ý: Nếu vẫn lỗi 'Cannot convert a symbolic Tensor', khả năng cao model này chứa Custom Layer không hỗ trợ TFLite.")