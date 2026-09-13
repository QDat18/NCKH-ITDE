"""
Streamlit Cloud Compatible Version
Deepfake Detection System - Video Lab Only
Removed webcam-dependent features for cloud deployment
"""
import streamlit as st
import cv2
import torch
import numpy as np
import os
import tempfile
import time
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path

# Add Demo directory to path
demo_dir = Path(__file__).parent / "Demo"
sys.path.insert(0, str(demo_dir))

from model_pytorch import DeepfakeEfficientNet
from collections import deque
from scipy.spatial.distance import cosine
import pandas as pd

MODEL_PATH = 'Demo/models/best_pytorch_model_final.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = (380, 380)

def load_face_cascade():
    """Load Haar Cascade for face detection"""
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    local_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Demo', 'haarcascade_frontalface_default.xml')
    
    if (not cascade_path or not os.path.exists(cascade_path)) and not os.path.exists(local_xml):
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            os.makedirs(os.path.dirname(local_xml), exist_ok=True)
            urllib.request.urlretrieve(url, local_xml)
        except Exception as e:
            st.warning(f"⚠️ Could not download cascade: {e}")
    
    cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else cv2.CascadeClassifier()
    if cascade.empty() and os.path.exists(local_xml):
        cascade = cv2.CascadeClassifier(local_xml)
    return cascade

face_cascade = load_face_cascade()

def safe_detect_faces(gray_img):
    """Safely detect faces with fallback"""
    if face_cascade is None or face_cascade.empty():
        h, w = gray_img.shape[:2]
        return np.array([[0, 0, w, h]])
    try:
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)
        return faces if len(faces) > 0 else np.array([[0, 0, gray_img.shape[1], gray_img.shape[0]]])
    except Exception:
        h, w = gray_img.shape[:2]
        return np.array([[0, 0, w, h]])

class IdentityTracker:
    """Track face identity and motion metrics"""
    def __init__(self, window_size=15):
        self.centroid_buffer = deque(maxlen=window_size)
        self.embed_buffer = deque(maxlen=window_size)
        self.risk_buffer = deque(maxlen=window_size)
        self.lap_buffer = deque(maxlen=window_size)
        
    def update(self, center, embed, raw_risk=0.0, lap_val=0.0):
        self.centroid_buffer.append(center)
        self.embed_buffer.append(embed)
        self.risk_buffer.append(raw_risk)
        self.lap_buffer.append(lap_val)
        
    def get_metrics(self, current_lap=None):
        drift = cosine(self.embed_buffer[-1], self.embed_buffer[-2]) if len(self.embed_buffer) > 1 else 0.0
        jitter = np.std(np.linalg.norm(np.diff(np.array(self.centroid_buffer), axis=0), axis=1)) if len(self.centroid_buffer) > 1 else 0.0
        smoothed_risk = np.median(self.risk_buffer) if len(self.risk_buffer) > 0 else 0.0
        
        z_lap = 0.0
        if current_lap is not None and len(self.lap_buffer) > 3:
            mean_lap = np.mean(self.lap_buffer)
            std_lap = np.std(self.lap_buffer) + 1e-6
            z_lap = (current_lap - mean_lap) / std_lap
            
        return drift, jitter, smoothed_risk, z_lap

def analyze_forensics(frame):
    """Forensic analysis: blur detection and FFT"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    fft = np.percentile(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1), 99) / (np.mean(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)) + 1e-6)
    
    # Calibration: reduced sensitivity for real webcams
    v_lap = 1 / (1 + np.exp(np.clip(0.4 * (lap - 8.0), -100, 100)))
    v_fft = 1 / (1 + np.exp(np.clip(-0.5 * (fft - 3.2), -100, 100)))
        
    return (v_lap + v_fft) / 2.0, lap

@st.cache_resource
def load_model():
    """Load pre-trained model"""
    model = DeepfakeEfficientNet(model_name='efficientnet_b4', pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        try:
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
            return model.to(DEVICE).eval(), 0.57
        except Exception as e:
            st.warning(f"⚠️ Could not load model: {e}")
            return model.to(DEVICE).eval(), 0.57
    else:
        st.warning(f"⚠️ Model weights not found at {MODEL_PATH}")
        return model.to(DEVICE).eval(), 0.57

def predict_expert(face_img, model):
    """Predict deepfake probability for a face image"""
    if face_img is None or face_img.size == 0:
        return 0.0, None
    
    tf = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = tf(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, embed = model(img, return_features=True)
        return torch.sigmoid(logits).item(), embed.cpu().numpy().flatten()

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="🛡️ Deepfake Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0c0a09; color: #f5f5f4; }
.expert-card { 
    background: #1c1917; 
    border-radius: 8px; 
    padding: 12px; 
    border: 1px solid #44403c;
}
.warning-box {
    background: #7c2d12;
    border-left: 4px solid #ea580c;
    padding: 12px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model, thresh = load_model()

# Sidebar configuration
with st.sidebar:
    st.title("🛡️ Secure Gateway")
    st.session_state.thresh = st.slider("Security Threshold (Ngưỡng bảo mật)", 0.1, 0.9, float(thresh))
    st.info("✅ Status: Deepfake Shield v8.0 Cloud Ready")
    st.markdown("---")
    st.markdown("""
    ### 📌 Cloud Version Info
    - ✅ Video Lab Analysis
    - ❌ Webcam: Not available on cloud
    - 📦 Model: EfficientNet-B4
    - 🔧 Framework: Streamlit
    """)

st.title("🛡️ Real-Time Deepfake Detection System")
st.markdown("**NCKH - Vietnam Banking Academy | EfficientNet-B4 Transfer Learning**")

st.warning("⚠️ **STREAMLIT CLOUD VERSION**: Webcam features are disabled. Use Video Lab to upload and analyze videos.", icon="⚠️")

# ============================================================================
# VIDEO LAB TAB - ONLY FUNCTIONAL ON CLOUD
# ============================================================================

st.markdown("---")
st.subheader("🔬 Video Lab - Upload & Analyze")
st.markdown("""
Upload a video file (MP4, MOV) to analyze for deepfake indicators:
- AI Detection Score
- Forensic Analysis (Blur, FFT)
- Identity Drift & Motion Jitter
- Statistical Report
""")

up_video = st.file_uploader("📤 Upload Video File", type=["mp4", "mov", "avi", "mkv"], label_visibility="visible")

if up_video:
    st.info(f"📹 Processing: {up_video.name} ({up_video.size / 1024 / 1024:.1f} MB)")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(up_video.read())
        tmp_path = tmp.name
    
    # Process video
    try:
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            st.error("❌ Cannot open video file. Try a different format.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.markdown(f"""
            **Video Info:**
            - Total Frames: {total_frames}
            - FPS: {fps:.1f}
            - Duration: {total_frames / fps:.1f}s
            """)
            
            scores = []
            drifts = []
            frames_processed = 0
            tr_lab = IdentityTracker()
            
            progress_bar = st.progress(0, text="Analyzing frames...")
            status_text = st.empty()
            
            frame_step = max(1, total_frames // 100)  # Analyze ~100 frames max
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for speed
                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue
                
                frame_idx += 1
                frames_processed += 1
                
                # Analyze frame
                f_resized = cv2.resize(frame, (640, 480))
                forensic_score, lap_val = analyze_forensics(f_resized)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = safe_detect_faces(gray)
                
                ai_score = 0.0
                if len(faces) > 0:
                    b = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
                    face_roi = frame[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
                    ai_score, em = predict_expert(face_roi, model)
                    
                    raw_risk = min(max(ai_score, forensic_score), 1.0)
                    tr_lab.update((b[0] + b[2]//2, b[1] + b[3]//2), em, raw_risk, lap_val)
                    drifts.append(tr_lab.get_metrics()[0])
                    scores.append(tr_lab.get_metrics()[2])
                else:
                    scores.append(forensic_score)
                    drifts.append(0.0)
                
                # Update progress
                progress = min(frame_idx / total_frames, 1.0)
                progress_bar.progress(progress, text=f"Analyzing {frames_processed} frames...")
            
            cap.release()
            progress_bar.empty()
            status_text.empty()
            
            # ================================================================
            # STATISTICAL ANALYSIS
            # ================================================================
            
            scores = np.array(scores)
            drifts = np.array(drifts)
            
            # Soft weighting for high-drift frames
            weights = np.ones_like(scores)
            weights[drifts > 0.85] = 0.3
            
            # Calculate statistics
            p95 = np.percentile(scores, 95) if len(scores) > 0 else 0
            mean = np.average(scores, weights=weights) if len(scores) > 0 else 0
            std = np.sqrt(np.average((scores - mean)**2, weights=weights)) if len(scores) > 0 else 0
            
            high_ratio = np.mean(scores > st.session_state.thresh) if len(scores) > 0 else 0
            spikes = np.sum(scores > 0.8) if len(scores) > 0 else 0
            d_p95 = np.percentile(drifts, 95) if len(drifts) > 0 else 0
            
            # Decision logic
            is_fake = False
            reasons = []
            
            if p95 > 0.75 and high_ratio > 0.15:
                is_fake = True
                reasons.append("High risk in many frames")
            
            if spikes > 3:
                is_fake = True
                reasons.append("Abnormal spikes detected")
            
            if d_p95 > 0.55:
                is_fake = True
                reasons.append("High identity drift")
            
            # Override: Accept low scores with stable behavior
            if mean < 0.4 and std < 0.15 and d_p95 <= 0.55:
                is_fake = False
                reasons = ["Stable behavior - Likely Real"]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Score", f"{mean:.4f}")
            with col2:
                st.metric("Std Dev", f"{std:.4f}")
            with col3:
                st.metric("P95", f"{p95:.4f}")
            with col4:
                st.metric("Drift P95", f"{d_p95:.4f}")
            
            # Verdict
            st.markdown("---")
            if is_fake:
                st.error(f"""
                ### 🚨 DEEPFAKE DETECTED
                **Verdict:** This video shows signs of deepfake/manipulation
                
                **Evidence:**
                {' | '.join(f'• {r}' for r in reasons)}
                
                **Metrics:**
                - P95 Score: {p95:.4f}
                - High-Risk Frames: {high_ratio:.1%}
                - Spikes (>0.8): {spikes}
                - Max Drift: {d_p95:.4f}
                """)
            else:
                st.success(f"""
                ### ✅ VIDEO AUTHENTIC
                **Verdict:** This video appears to be genuine
                
                **Evidence:**
                {' | '.join(f'• {r}' for r in reasons)}
                
                **Metrics:**
                - Mean Score: {mean:.4f}
                - Std Dev: {std:.4f}
                - P95: {p95:.4f}
                - Stability: Stable
                """)
            
            # Chart
            st.markdown("---")
            col_chart, col_hist = st.columns(2)
            
            with col_chart:
                st.line_chart(scores, use_container_width=True)
                st.caption("Risk Score Over Time")
            
            with col_hist:
                st.bar_chart(pd.Series(scores).value_counts().sort_index(), use_container_width=True)
                st.caption("Score Distribution")
            
            # Detailed stats table
            st.markdown("---")
            st.subheader("📈 Detailed Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': [
                    'Mean Score',
                    'Std Deviation',
                    'P95 Score',
                    'High-Risk Ratio',
                    'Spike Count (>0.8)',
                    'Max Drift',
                    'Frames Analyzed',
                    'Total Frames'
                ],
                'Value': [
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    f"{p95:.6f}",
                    f"{high_ratio:.2%}",
                    f"{int(spikes)}",
                    f"{d_p95:.6f}",
                    f"{len(scores)}",
                    f"{total_frames}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📚 About This Project
- **Institution**: Vietnam Banking Academy (NCKH)
- **Author**: Quốc Đạt
- **Email**: qdat18@gmail.com
- **Model**: EfficientNet-B4 with Transfer Learning
- **Framework**: Streamlit + PyTorch

⭐ **Cloud Version**: Limited to Video Lab (offline analysis)
🖥️ **Local Version**: Full features including webcam & real-time detection
""")

st.markdown("""
**License**: MIT | [GitHub](https://github.com/QDat18/NCKH-ITDE)
""")
