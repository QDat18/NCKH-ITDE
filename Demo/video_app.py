import streamlit as st
import cv2
import torch
import numpy as np
import os
import tempfile
import time
import threading
import queue
from PIL import Image
from torchvision import transforms
from model_pytorch import DeepfakeEfficientNet
from collections import deque
from scipy.spatial.distance import cosine
import pandas as pd

MODEL_PATH = 'models/best_pytorch_model_final.pth' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = (380, 380)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class IdentityTracker:
    def __init__(self, window_size=15):
        self.centroid_buffer = deque(maxlen=window_size)
        self.embed_buffer = deque(maxlen=window_size)
        self.risk_buffer = deque(maxlen=window_size) # For temporal voting
        self.lap_buffer = deque(maxlen=window_size)  # For Statistical Laplacian Z-score
        
    def update(self, center, embed, raw_risk=0.0, lap_val=0.0):
        self.centroid_buffer.append(center); self.embed_buffer.append(embed)
        self.risk_buffer.append(raw_risk)
        self.lap_buffer.append(lap_val)
        
    def get_metrics(self, current_lap=None):
        drift = cosine(self.embed_buffer[-1], self.embed_buffer[-2]) if len(self.embed_buffer)>1 else 0.0
        jitter = np.std(np.linalg.norm(np.diff(np.array(self.centroid_buffer), axis=0), axis=1)) if len(self.centroid_buffer)>1 else 0.0
        smoothed_risk = np.median(self.risk_buffer) if len(self.risk_buffer) > 0 else 0.0
        
        # Calculate dynamic laplacian z-score
        z_lap = 0.0
        if current_lap is not None and len(self.lap_buffer) > 3:
            mean_lap = np.mean(self.lap_buffer)
            std_lap = np.std(self.lap_buffer) + 1e-6
            z_lap = (current_lap - mean_lap) / std_lap
            
        return drift, jitter, smoothed_risk, z_lap

class AsyncStream:
    def __init__(self, source):
        self.source = source; self.cap = cv2.VideoCapture(source); self.q = queue.Queue(maxsize=1)
        self.stopped = False; self.thread = threading.Thread(target=self._update, daemon=True)
    def start(self): self.thread.start(); return self
    def stop(self): self.stopped = True; self.cap.release()
    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str): self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                self.stop(); break
            if self.q.full():
                try: self.q.get_nowait()
                except: pass
            self.q.put(frame)
    def read(self):
        try: return True, self.q.get(timeout=1.0)
        except: return False, None

def analyze_forensics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    fft = np.percentile(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1), 99) / (np.mean(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)) + 1e-6)
    
    # Hiệu chuẩn v7.0: Giảm độ nhạy với Webcams thật (giảm False Positive)
    # 1. Blur Tolerance: Chấp nhận độ mờ tự nhiên (lap > 8 thay vì 15)
    v_lap = 1 / (1 + np.exp(np.clip(0.4 * (lap - 8.0), -100, 100)))
    
    # 2. Noise Tolerance: Chấp nhận hạt nhiễu (noise) của camera rẻ tiền (fft < 3.2 thay vì 2.4)
    v_fft = 1 / (1 + np.exp(np.clip(-0.5 * (fft - 3.2), -100, 100)))
        
    return (v_lap + v_fft) / 2.0, lap

@st.cache_resource
def load_model():
    model = DeepfakeEfficientNet(model_name='efficientnet_b4', pretrained=False)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) else ckpt)
    return model.to(DEVICE).eval(), 0.57

def predict_expert(face_img, model):
    if face_img is None or face_img.size == 0: return 0.0, None
    tf = transforms.Compose([transforms.Resize(TARGET_SIZE), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img = tf(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, embed = model(img, return_features=True)
        return torch.sigmoid(logits).item(), embed.cpu().numpy().flatten()

st.set_page_config(page_title="Hệ thống phát hiện Deepfake NCKH", layout="wide")
st.markdown("<style>.stApp { background-color: #0c0a09; color: #f5f5f4; } .expert-card { background: #1c1917; border-radius: 8px; padding: 12px; border: 1px solid #44403c; }</style>", unsafe_allow_html=True)

model, thresh = load_model()
with st.sidebar:
    st.title("🛡️ Secure Gateway")
    st.session_state.thresh = st.slider("Ngưỡng bảo mật", 0.1, 0.9, float(thresh))
    st.info("Trạng thái: Deepfake Shield v7.0 Đang Hoạt Động")

st.title("Hệ thống phát hiện deepfake sử dụng EfficientNet và Học chuyển giao")
tabs = st.tabs(['Đối chứng Thực nghiệm', 'Xác thực Sinh trắc (eKYC)', 'Webcam', 'Kiểm thử Video'])
t_sim, t_call, t_mon, t_lab = tabs

with t_sim:
    st.markdown("<div class='expert-card' style='text-align:center;'><b>⚔️ HỆ THỐNG ĐỐI KHÁNG KÉP (ADVERSARIAL SIMULATOR)</b><br><small>So sánh trực tiếp độ tin cậy sinh học giữa Webcam và Video tải lên.</small></div>", unsafe_allow_html=True)
    
    col_ctrl, col_alr = st.columns([1.5, 2])
    with col_ctrl:
        src_opt_sim = st.radio("Nguồn Camera chính:", ["📷 Webcam Máy tính", "🎥 OBS Virtual Camera"])
        cam_src = 0
        if src_opt_sim == "🎥 OBS Virtual Camera":
            cam_src = st.number_input("ID Camera OBS (Thử 1, 2, 3... nếu không lên hình)", min_value=0, max_value=10, value=1, key="obs_sim")
        up = st.file_uploader("Nạp Video Tấn công (Attack Payload)", type=["mp4","mov"], label_visibility="collapsed")
        run = st.button("⚡ KHỞI CHẠY ĐỐI KHÁNG", type="primary", use_container_width=True)
    with col_alr:
        alr = st.empty()

    if run and up:
        canv = st.empty()
        c_m1, c_m2 = st.columns(2)
        with c_m1: h1 = st.empty()
        with c_m2: h2 = st.empty()
        
        t = tempfile.NamedTemporaryFile(delete=False); t.write(up.read())
        s1, s2 = AsyncStream(cam_src).start(), AsyncStream(t.name).start()
        tr1, tr2 = IdentityTracker(), IdentityTracker()
        
        while not s1.stopped and not s2.stopped:
            ok1, f1 = s1.read(); ok2, f2 = s2.read()
            if not ok1 or not ok2: continue
            f1, f2 = cv2.resize(cv2.flip(f1, 1), (500, 400)), cv2.resize(f2, (500, 400))
            ovrs = []
            
            for idx, (frame, tracker) in enumerate([(f1, tr1), (f2, tr2)]):
                fr, lap_val = analyze_forensics(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 6)
                p, em = 0.0, None
                if len(faces) > 0:
                    b = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                    p, em = predict_expert(frame[b[1]:b[1]+b[3], b[0]:b[0]+b[2]], model)
                    
                    # Statistical Z-score check
                    drift, jitter, smoothed_risk, z_lap = tracker.get_metrics(current_lap=lap_val)
                    
                    # Uncertainty logic (Z-score > 2.5 means sudden anomaly, or completely out of domain)
                    unc_penalty = 0.15 if abs(z_lap) > 2.5 else 0.0
                    is_unc = True if unc_penalty > 0 else False
                    
                    # Robust Fusion Update
                    raw_risk = min(max(p, fr) + unc_penalty, 1.0)
                    tracker.update((b[0]+b[2]//2, b[1]+b[3]//2), em, raw_risk, lap_val)
                    
                    # Get final metrics after update
                    drift, jitter, risk, _ = tracker.get_metrics()
                    
                    if drift > 0.50: risk = max(risk, 0.85); ovrs.append("Lệch định danh")
                    if jitter > 25.0: risk = max(risk, 0.82); ovrs.append("Rung lắc pixel")
                    if is_unc: ovrs.append(f"Tín hiệu nhiễu/mờ (Z={z_lap:.1f})")
                    
                    is_fake = risk > st.session_state.thresh
                    color = (0,0,255) if is_fake else (0,255,0)
                    cv2.rectangle(frame, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), color, 2)
                    
                    # Trình bày UI chuyên nghiệp
                    s_name = "🟢 LUỒNG WEBCAM (LIVE)" if idx==0 else "🔴 LUỒNG TẤN CÔNG (PAYLOAD)"
                    status = "CẢNH BÁO GIẢ MẠO" if is_fake else "XÁC MINH AN TOÀN"
                    b_col = "#ef4444" if is_fake else "#22c55e"
                    
                    h = h1 if idx==0 else h2
                    h.markdown(f"""
                    <div class='expert-card' style='border-top: 4px solid {b_col};'>
                        <h4 style='margin-top:0px; color:{b_col};'>{s_name}</h4>
                        <div style='display:flex; justify-content:space-between; border-bottom: 1px solid #44403c; padding-bottom:4px; margin-bottom:4px;'><span>Trạng thái:</span> <b>{status}</b></div>
                        <div style='display:flex; justify-content:space-between;'><span>Rủi ro tổng thể:</span> <b>{risk:.2%}</b></div>
                        <div style='display:flex; justify-content:space-between;'><span>Identity Drift:</span> <b>{drift:.3f}</b></div>
                        <div style='display:flex; justify-content:space-between;'><span>Motion Jitter:</span> <b>{jitter:.1f}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if ovrs: alr.error(f"🚨 TƯỜNG LỬA CHỈ ĐIỂM: Bắt được nỗ lực che giấu ({' | '.join(list(set(ovrs)))})")
            else: alr.info("ℹ️ GIÁM SÁT THỜI GIAN THỰC: Cả hai luồng đang trong ngưỡng an toàn sinh học.")
            
            # Gạch phân cách
            cv2.line(f1, (499, 0), (499, 400), (80, 80, 80), 3)
            canv.image(cv2.cvtColor(np.hstack((f1, f2)), cv2.COLOR_BGR2RGB))
            
        s1.stop(); s2.stop()

with t_call:
    st.markdown("<div class='expert-card'><b>🛡️ CỔNG ĐỊNH DANH SINH TRẮC HỌC X-eKYC</b><br><small>Mô phỏng quy trình định danh khách hàng. Bạn có thể chèn video giả mạo để thử nghiệm khả năng phòng thủ dưới góc độ ngân hàng.</small></div>", unsafe_allow_html=True)
    if 'call_on' not in st.session_state: st.session_state.call_on = False
    
    if not st.session_state.call_on:
        src_opt = st.radio("Chọn Nguồn Camera Đầu vào (Camera Source):", ["📷 Webcam Máy tính", "🎥 OBS Virtual Camera"])
        cam_src_call = 0
        if src_opt == "🎥 OBS Virtual Camera":
            cam_src_call = st.number_input("ID Camera OBS (Thử 1, 2, 3...)", min_value=0, max_value=10, value=1, key="obs_call")
            
        if st.button("KÍCH HOẠT KÊNH AN TOÀN", use_container_width=True, type="primary"):
            st.session_state.call_source = cam_src_call
            st.session_state.call_on = True; st.rerun()
    else:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔴 NGẮT KẾT NỐI", use_container_width=True, key="stop_secure_call"):
                st.session_state.call_on = False; st.rerun()
        with col_btn2:
            simulate_attack = st.toggle("🦹 Bật Simulation Attack Mode (Spoofing)", key="sim_atk")
            
        col_v, col_m = st.columns([2, 1])
        with col_v:
            cnv_c = st.empty()
        with col_m:
            status_box = st.empty()
            
        strm_c = AsyncStream(st.session_state.call_source).start()
        tr_c = IdentityTracker() # Thêm tracker cho Call
        
        while st.session_state.call_on and not strm_c.stopped:
            ok, fc = strm_c.read()
            if not ok: continue
            fc = cv2.resize(cv2.flip(fc, 1), (640, 480))
            
            gray_initial = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_initial, 1.1, 6)
            
            # Hacker Attack Simulation (FaceSwap)
            if simulate_attack and len(faces) > 0:
                b = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                try:
                    mask = cv2.imread("hacker_mask.png")
                    if mask is not None:
                        mask = cv2.resize(mask, (b[2], b[3]))
                        fc[b[1]:b[1]+b[3], b[0]:b[0]+b[2]] = mask
                        cv2.putText(fc, "SPOOFING ACTIVE", (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                except:
                    pass

            # Phân tích đa luồng
            fr, lap_val = analyze_forensics(fc)
            ai_s, drift, jitter, smoothed_risk = 0.0, 0.0, 0.0, 0.0
            is_unc = False
            if len(faces) > 0:
                b = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                ai_s, em = predict_expert(fc[b[1]:b[1]+b[3], b[0]:b[0]+b[2]], model)
                
                _, _, _, z_lap = tr_c.get_metrics(current_lap=lap_val)
                unc_penalty = 0.15 if abs(z_lap) > 2.5 else 0.0
                is_unc = True if unc_penalty > 0 else False
                
                raw_risk = min(max(ai_s, fr) + unc_penalty, 1.0)
                tr_c.update((b[0]+b[2]//2, b[1]+b[3]//2), em, raw_risk, lap_val)
                drift, jitter, smoothed_risk, _ = tr_c.get_metrics()
            
            risk = smoothed_risk
            if drift > 0.50: risk = max(risk, 0.85)
            
            status_text = "CẢNH BÁO KÉM CHẤT LƯỢNG" if is_unc and risk <= st.session_state.thresh else ("AN TOÀN" if risk <= st.session_state.thresh else "CẢNH BÁO GIẢ MẠO")
            border_color = "#eab308" if (is_unc and risk <= st.session_state.thresh) else ("#22c55e" if risk <= st.session_state.thresh else "#ef4444")
            
            status_box.markdown(f"""
            <div class='expert-card' style='border-left: 5px solid {border_color}'>
                <b>TRẠNG THÁI: {status_text}</b><br>
                Định danh: {"Đã xác minh" if risk < st.session_state.thresh else "Nghi vấn giả mạo"}<br>
                Rủi ro tổng hợp: {risk:.2%}<br>
                <hr style='margin: 8px 0; border-color: #44403c'>
                📈 Tiêu chuẩn Drift: <b>{drift:.3f}</b><br>
                📉 Tiêu chuẩn Jitter: <b>{jitter:.1f}</b>
            </div>
            """, unsafe_allow_html=True)
            
            cnv_c.image(cv2.cvtColor(fc, cv2.COLOR_BGR2RGB))
            
        strm_c.stop()

with t_mon:
    src_opt_mon = st.radio("Chọn Camera Giám sát:", ["📷 Webcam Máy tính", "🎥 OBS Virtual Camera"])
    cam_src_mon = 0
    if src_opt_mon == "🎥 OBS Virtual Camera":
        cam_src_mon = st.number_input("ID Camera OBS (Thử 1, 2, 3...)", min_value=0, max_value=10, value=1, key="obs_mon")
    if st.button("BẮT ĐẦU GIÁM SÁT"):
        strm_m = AsyncStream(cam_src_mon).start(); cnv_m = st.empty()
        while not strm_m.stopped:
            ok, fm = strm_m.read()
            if not ok: continue
            fm = cv2.resize(cv2.flip(fm, 1), (640, 480))
            cons, _ = analyze_forensics(fm)
            color = (0,0,255) if cons > st.session_state.thresh else (0,255,0)
            cv2.putText(fm, f"RISK: {cons:.2%}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cnv_m.image(cv2.cvtColor(fm, cv2.COLOR_BGR2RGB))
        strm_m.stop()

with t_lab:
    up2 = st.file_uploader("Tải lên tệp bằng chứng", type=["mp4","mov"])
    if up2:
        tf = tempfile.NamedTemporaryFile(delete=False); tf.write(up2.read())
        cap = cv2.VideoCapture(tf.name); scores, drifts = [], []
        pb = st.progress(0, text="Quét tệp bằng chứng...")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); curr = 0; tr3 = IdentityTracker()
        while cap.isOpened():
            ret, f = cap.read()
            if not ret: break
            curr += 1
            if curr % 15 != 0: continue
            f_r, lap_val = analyze_forensics(cv2.resize(f, (640, 480)))
            faces = face_cascade.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 1.1, 6)
            a_s = 0.0
            if len(faces) > 0:
                b = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                a_s, em = predict_expert(f[b[1]:b[1]+b[3], b[0]:b[0]+b[2]], model)
                
                # We skip deep Z-score in lab mode, because it's non-continuous
                # but we still update it. Let's use a relaxed penalty if variance drops fast
                unc = 0.0
                raw_r = min(max(a_s, f_r) + unc, 1.0)
                tr3.update((0,0), em, raw_r, lap_val)
                drifts.append(tr3.get_metrics()[0])
                scores.append(tr3.get_metrics()[2])
            else:
                scores.append(f_r)
                drifts.append(0.0) # Pad drifts để cân bằng mảng
            pb.progress(min(curr/total, 1.0))
        cap.release(); st.line_chart(scores)
        
        # --- TIẾN TRÌNH ĐÁNH GIÁ THỐNG KÊ (HEURISTICS) ---
        scores = np.array(scores)
        drifts = np.array(drifts)

        # 1. Soft filter thay vì hard remove
        weights = np.ones_like(scores)
        weights[drifts > 0.85] = 0.3  # giảm trọng số, không xóa

        # 2. Stats
        p95 = np.percentile(scores, 95) if len(scores) > 0 else 0
        mean = np.average(scores, weights=weights) if len(scores) > 0 else 0
        std = np.sqrt(np.average((scores - mean)**2, weights=weights)) if len(scores) > 0 else 0

        high_ratio = np.mean(scores > st.session_state.thresh) if len(scores) > 0 else 0
        spikes = np.sum(scores > 0.8) if len(scores) > 0 else 0

        # Drift stats
        d_p95 = np.percentile(drifts, 95) if len(drifts) > 0 else 0

        # 3. Decision logic
        is_fake = False
        reason = []

        if p95 > 0.75 and high_ratio > 0.15:
            is_fake = True
            reason.append("Nhiều frame có rủi ro cao")

        if spikes > 3:
            is_fake = True
            reason.append("Xuất hiện spike bất thường")

        if d_p95 > 0.55:
            is_fake = True
            reason.append("Lệch định danh cao")

        # OVERRIDER: Chỉ 'đại xá' nhiễu AI nếu KHÔNG bị Lệch định danh
        if mean < 0.4 and std < 0.15 and d_p95 <= 0.55:
            is_fake = False
            reason = ["Hành vi ổn định, giống người thật"]

        # 4. Output
        if is_fake:
            st.error(f"🚨 **PHÁT HIỆN GIẢ MẠO (DEEPFAKE)**\n\n- Chỉ số P95: **{p95:.4f}**\n- Điểm trung bình: **{mean:.4f}**\n- Độ phân tán (Std): **{std:.4f}**\n- Số lần vọt (Spikes): **{spikes}**\n- Lệch định danh P95: **{d_p95:.4f}**\n\n🔎 **Lý do:** {', '.join(reason)}")
        else:
            st.success(f"✅ **VIDEO TIN CẬY (NGƯỜI THẬT)**\n\n- Điểm trung bình: **{mean:.4f}**\n- Độ phân tán (Std): **{std:.4f}**\n- Chỉ số P95: **{p95:.4f}**\n\n📊 Hành vi sinh học nhất quán và ổn định theo thời gian.")