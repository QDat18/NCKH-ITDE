import cv2
import pyvirtualcam
import os
import sys
import time

def run_simulator(video_path):
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    print(f"--- Attacker Simulator Started ---")
    print(f"Video: {video_path} ({width}x{height} @ {fps}fps)")
    print(f"Streaming to Virtual Camera... Press Ctrl+C to stop.")

    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
            print(f"Virtual Camera Active: {cam.device}")
            
            while True:
                ret, frame = cap.read()
                
                # Loop the video if it reaches the end
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Convert BGR (OpenCV) to RGB (Virtual Cam)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Send frame
                cam.send(frame_rgb)
                cam.sleep_until_next_frame()
                
    except KeyboardInterrupt:
        print("\nSimulator stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    # Check for command line argument or use default
    target_video = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not target_video:
        print("Usage: python Simulator_FakeCall.py <path_to_video>")
        print("\nTIP: Since no video was provided, checking for common files in workspace...")
        # Check for any .mp4 files if none specified
        mp4_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
        if mp4_files:
            print(f"Found {mp4_files[0]}. Starting simulation with it...")
            run_simulator(mp4_files[0])
        else:
            print("No video files found. Please provide a path to a deepfake video.")
    else:
        run_simulator(target_video)
