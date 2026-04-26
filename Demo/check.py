import mediapipe as mp
try:
    print("MediaPipe Version:", mp.__version__)
    print("Solutions available:", hasattr(mp, 'solutions'))
    face = mp.solutions.face_detection
    print("✅ MediaPipe hoạt động bình thường!")
except Exception as e:
    print(f"❌ Vẫn lỗi: {e}")