"""
Streamlit Cloud Compatible Version
Deepfake Detection System - Video Lab Only
Uses PIL instead of OpenCV to avoid libGL dependency
"""
import streamlit as st
import numpy as np
import os
import tempfile
from PIL import Image
import torch
from torchvision import transforms
from collections import deque
from scipy.spatial.distance import cosine
import pandas as pd
import gdown

MODEL_PATH = 'Demo/models/best_pytorch_model_final.pth'
GOOGLE_DRIVE_ID = "1MkgcU0iAlBT3B0aSdxe57RdV1qF6zpa5"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = (380, 380)

st.set_page_config(
    page_title="🛡️ Deepfake Detection System",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

# Import model
import sys
from pathlib import Path
demo_dir = Path(__file__).parent / "Demo"
sys.path.insert(0, str(demo_dir))

try:
    from model_pytorch import DeepfakeEfficientNet
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.stop()

def download_model_from_gdrive():
    """Download model from Google Drive"""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model weights (first run ~2-3 min)...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}",
                MODEL_PATH,
                quiet=False
            )
            st.success("✅ Model downloaded!")
            return True
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            return False
    return True

@st.cache_resource
def load_model():
    """Load model"""
    if not download_model_from_gdrive():
        return None, 0.57
    
    model = DeepfakeEfficientNet(model_name='efficientnet_b4', pretrained=False)
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
        return model.to(DEVICE).eval(), 0.57
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return model.to(DEVICE).eval(), 0.57

class IdentityTracker:
    """Track metrics"""
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

def analyze_image_pil(img_pil):
    """Analyze image using PIL (no OpenCV/libGL needed)"""
    # Convert to numpy
    img_array = np.array(img_pil.convert('L'))  # Grayscale
    
    # Laplacian variance (blur detection)
    from scipy import ndimage
    laplacian = ndimage.laplace(img_array)
    lap_var = np.var(laplacian)
    
    # FFT analysis
    fft = np.abs(np.fft.fft2(img_array))
    fft_log = np.log(fft + 1)
    fft_score = np.percentile(fft_log, 99) / (np.mean(fft_log) + 1e-6)
    
    # Sigmoid calibration
    v_lap = 1 / (1 + np.exp(np.clip(0.4 * (lap_var - 8.0), -100, 100)))
    v_fft = 1 / (1 + np.exp(np.clip(-0.5 * (fft_score - 3.2), -100, 100)))
    
    return (v_lap + v_fft) / 2.0, lap_var

def predict_expert(face_img_pil, model):
    """Predict using PIL image"""
    if face_img_pil is None:
        return 0.0, None
    
    face_img_pil = face_img_pil.resize(TARGET_SIZE)
    
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = tf(face_img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, embed = model(img, return_features=True)
        return torch.sigmoid(logits).item(), embed.cpu().numpy().flatten()

# UI
model, thresh = load_model()

if model is None:
    st.error("❌ Cannot load model")
    st.stop()

with st.sidebar:
    st.title("🛡️ Secure Gateway")
    st.session_state.thresh = st.slider("Security Threshold", 0.1, 0.9, float(thresh))
    st.info("✅ Deepfake Shield v9.0 Cloud Ready")

st.title("🛡️ Real-Time Deepfake Detection System")
st.markdown("**NCKH - Vietnam Banking Academy**")
st.warning("⚠️ **STREAMLIT CLOUD VERSION**: Video Lab only", icon="⚠️")

st.markdown("---")
st.subheader("🔬 Video Lab - Upload & Analyze")

up_video = st.file_uploader("📤 Upload Video", type=["mp4", "mov", "avi"])

if up_video:
    st.info(f"📹 Processing: {up_video.name}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(up_video.read())
        tmp_path = tmp.name
    
    try:
        # Use imageio instead of cv2
        import imageio
        
        reader = imageio.get_reader(tmp_path)
        total_frames = reader.count_frames()
        
        st.markdown(f"**Video:** {total_frames} frames")
        
        scores = []
        drifts = []
        tr_lab = IdentityTracker()
        progress_bar = st.progress(0, text="Analyzing...")
        
        frame_step = max(1, total_frames // 50)
        frame_idx = 0
        
        for frame_num, frame in enumerate(reader):
            if frame_num % frame_step != 0:
                continue
            
            frame_idx += 1
            
            # Convert frame to PIL
            frame_pil = Image.fromarray(frame)
            frame_resized = frame_pil.resize((640, 480))
            
            forensic_score, lap_val = analyze_image_pil(frame_resized)
            
            # Simple face detection using PIL (no cascade needed)
            # For cloud, we just use the full frame
            face_roi = frame_resized.crop((50, 50, 590, 430))
            
            ai_score, em = predict_expert(face_roi, model)
            raw_risk = min(max(ai_score, forensic_score), 1.0)
            tr_lab.update((320, 240), em, raw_risk, lap_val)
            
            drifts.append(tr_lab.get_metrics()[0])
            scores.append(tr_lab.get_metrics()[2])
            
            progress_bar.progress(min(frame_idx / max(total_frames // frame_step, 1), 1.0))
        
        progress_bar.empty()
        
        scores = np.array(scores)
        drifts = np.array(drifts)
        weights = np.ones_like(scores)
        weights[drifts > 0.85] = 0.3
        
        p95 = np.percentile(scores, 95) if len(scores) > 0 else 0
        mean = np.average(scores, weights=weights) if len(scores) > 0 else 0
        std = np.sqrt(np.average((scores - mean)**2, weights=weights)) if len(scores) > 0 else 0
        high_ratio = np.mean(scores > st.session_state.thresh) if len(scores) > 0 else 0
        spikes = np.sum(scores > 0.8)
        d_p95 = np.percentile(drifts, 95) if len(drifts) > 0 else 0
        
        is_fake = False
        reasons = []
        
        if p95 > 0.75 and high_ratio > 0.15:
            is_fake = True
            reasons.append("High risk frames")
        if spikes > 3:
            is_fake = True
            reasons.append("Spikes detected")
        if d_p95 > 0.55:
            is_fake = True
            reasons.append("High drift")
        
        if mean < 0.4 and std < 0.15 and d_p95 <= 0.55:
            is_fake = False
            reasons = ["Stable - Authentic"]
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Mean", f"{mean:.4f}")
        with col2: st.metric("Std", f"{std:.4f}")
        with col3: st.metric("P95", f"{p95:.4f}")
        with col4: st.metric("Drift", f"{d_p95:.4f}")
        
        st.markdown("---")
        if is_fake:
            st.error(f"### 🚨 DEEPFAKE\n\n{' | '.join(reasons)}")
        else:
            st.success(f"### ✅ AUTHENTIC\n\n{' | '.join(reasons)}")
        
        st.line_chart(scores)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.markdown("---")
st.markdown("**NCKH** | [GitHub](https://github.com/QDat18/NCKH-ITDE)")
