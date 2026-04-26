import pyvirtualcam
import torch
import mediapipe as mp
import cv2

def check_env():
    print("--- Environment Check ---")
    
    # 1. Check Torch & CUDA
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Check Face Detection (OpenCV)
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not cascade.empty():
            print("Face Detection (OpenCV): OK")
        else:
            print("Face Detection (OpenCV): FAILED (Missing XML)")
    except Exception as e:
        print(f"Face Detection (OpenCV): Error - {e}")

    
    # 3. Check Virtual Camera
    print("\n--- Virtual Camera Check ---")
    try:
        with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
            print(f"Virtual Camera Device Found: {cam.device}")
            print("Status: READY")
    except Exception as e:
        print(f"Virtual Camera: NOT FOUND or ERROR")
        print(f"Error Details: {e}")
        print("\nTIP: You might need to install OBS (obsproject.com) to get the 'OBS Virtual Camera' driver.")

if __name__ == "__main__":
    check_env()
    print("\nIf all 'OK' and 'READY', you can run 'python Video_call_virtual.py'")
