# 🛡️ Real-Time Deepfake Detection System for Secure Video Calls

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-ff0000.svg)](https://streamlit.io/)
[![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B4-green.svg)](https://arxiv.org/abs/1905.11946)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hệ thống phát hiện Deepfake theo thời gian thực dành cho các ứng dụng bảo mật trong video conferencing và eKYC (Electronic Know Your Customer). Dự án sử dụng **EfficientNet-B4** kết hợp với **Transfer Learning** và các kỹ thuật phân tích pháp y để đạt độ chính xác cao.

---

## 🎯 Mục đích Dự án

Đây là dự án **Nghiên cứu Khoa học (NCKH)** của Học viện Ngân hàng Việt Nam nhằm:
- Phát hiện giả mạo video (Deepfake) trong thời gian thực
- Hỗ trợ xác thực sinh trắc học (eKYC) an toàn
- Cảnh báo các cuộc tấn công spoofing và lợi dụng mặt nạ

---

## 🚀 Tính năng Chính

### 1. **Đối chứng Thực nghiệm (Adversarial Simulator)**
- So sánh song song luồng Webcam trực tiếp với video tấn công (Attack Payload)
- Kiểm tra độ tin cậy sinh học của camera chính
- Phát hiện kỹ thuật lợi dụng mặt nạ (mask spoofing), deepfake, face swap

### 2. **Xác thực Sinh trắc (eKYC)**
- Mô phỏng quy trình định danh khách hàng trực tuyến
- Phát hiện các tín hiệu giả mạo trong thời gian thực
- Chế độ mô phỏng tấn công (Spoofing Simulation)

### 3. **Giám sát Webcam (Live Monitoring)**
- Theo dõi camera trong thời gian thực
- Cảnh báo tức thời khi phát hiện rủi ro Deepfake
- Hiển thị chỉ số rủi ro trực tiếp trên video

### 4. **Kiểm thử Video Ngoại tuyến (Video Lab)**
- Phân tích tệp video được tải lên
- Báo cáo chi tiết thống kê:
  - **P95**: Giá trị tại phần trăm 95
  - **Mean**: Điểm trung bình
  - **Std**: Độ phân tán chuẩn
  - **Spikes**: Số lần vượt ngưỡng 0.8
  - **Identity Drift**: Sự thay đổi định danh
- Biểu đồ và trực quan hóa chi tiết

---

## 📊 Hiệu năng & Kết quả

### Các Chỉ số Phân loại

| Chỉ số | Giá trị |
| :--- | :--- |
| **Accuracy** | >96% |
| **AUC Score** | 0.94 |
| **Inference Time** | 35-50ms/frame |
| **FPS** | 25-30 FPS |
| **Stability** | Cao (Temporal Window) |

### Phương pháp Đánh giá

Hệ thống kết hợp nhiều yếu tố:

1. **AI Prediction Score**: Điểm từ mô hình Deep Learning (EfficientNet-B4)
2. **Forensic Analysis**:
   - **Laplacian Variance**: Phát hiện độ mờ bất thường
   - **FFT Analysis**: Phân tích tần số để phát hiện artifacts
3. **Temporal Smoothing**: Sliding Window (k=10) để ổn định dự đoán
4. **Identity Drift**: Theo dõi sự thay đổi định danh giữa các frame
5. **Motion Jitter**: Phân tích dao động bất thường của vùng mặt

---

## 🏗️ Kiến trúc Hệ thống

```
Input Stream (Webcam/Video)
    ↓
[Face Detection] - MTCNN & Haar Cascade
    ↓
[Preprocessing]
  - Face Extraction
  - 15% Contextual Padding
  - Laplacian Blur Check
    ↓
[Feature Extraction] - EfficientNet-B4 Backbone
    ↓
[Classification Head]
  - RGB Projection (512 dims)
  - Binary Classification (Real/Fake)
    ↓
[Post-Processing]
  - Temporal Smoothing
  - Identity Drift Calculation
  - Motion Jitter Analysis
    ↓
Output (Real/Fake + Risk Score)
```

---

## 🛠️ Tech Stack

### Core Libraries
- **Ngôn ngữ**: Python 3.9+
- **Deep Learning**: PyTorch, Torchvision, TIMM
- **Framework Web**: Streamlit
- **Computer Vision**: OpenCV, MTCNN, MediaPipe
- **Data Science**: NumPy, SciPy, Pandas, Matplotlib

### Kiến trúc Mô hình
- **Base**: EfficientNet-B4 (ImageNet Pretrained)
- **Input Size**: 380×380×3
- **Classification Head**: Binary (Real/Fake)
- **Optional**: Frequency Artifact Branch

---

## 📁 Cấu trúc Dự án

```
NCKH-ITDE/
├── Demo/
│   ├── video_app.py              # 🎯 Ứng dụng chính (Streamlit UI)
│   ├── model_pytorch.py          # Định nghĩa kiến trúc EfficientNet-B4
│   ├── requirement.txt           # Thư viện Demo
│   ├── README.md                 # Hướng dẫn Chi tiết
│   ├── hacker_mask.png           # Tài nguyên mô phỏng tấn công
│   └── models/                   # Thư mục chứa weights
│       └── best_pytorch_model_final.pth
│
├── crop_face.py                  # Tiện ích trích xuất khuôn mặt
├── requirements.txt              # Thư viện chính
├── README.md                     # File này
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. **Yêu cầu Hệ thống**
- Python 3.9 hoặc cao hơn
- 4GB RAM tối thiểu (8GB khuyên cáo)
- GPU NVIDIA (tùy chọn, để tăng tốc độ)
- Webcam hoặc video file

### 2. **Clone Repository**

```bash
git clone https://github.com/QDat18/NCKH-ITDE.git
cd NCKH-ITDE
```

### 3. **Cài đặt Môi trường Ảo**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 4. **Cài đặt Thư viện**

```bash
# Cài đặt thư viện chính
pip install -r requirements.txt

# Cài đặt thư viện Demo (trong thư mục Demo)
cd Demo
pip install -r requirement.txt
```

### 5. **Chuẩn bị Mô hình**

Đảm bảo file weights `best_pytorch_model_final.pth` được đặt tại:
```
Demo/models/best_pytorch_model_final.pth
```

> **Lưu ý**: File model weights lớn (>500MB). Vui lòng liên hệ tác giả để nhận access link.

### 6. **Khởi chạy Ứng dụng**

```bash
cd Demo
streamlit run video_app.py
```

Ứng dụng sẽ mở tự động tại: `http://localhost:8501`

---

## 💻 Hướng dẫn Sử dụng

### Tab 1: Đối chứng Thực nghiệm (Adversarial Simulator)
1. Chọn nguồn camera (Webcam hoặc OBS Virtual Camera)
2. Tải lên video tấn công (attack payload)
3. Nhấn "KHỞI CHẠY ĐỐI KHÁNG"
4. Quan sát kết quả song song của 2 luồng video

### Tab 2: Xác thực Sinh trắc (eKYC)
1. Chọn camera nguồn
2. Nhấn "KÍCH HOẠT KÊNH AN TOÀN"
3. Tùy chọn bật "Spoofing Simulation Mode"
4. Xem trạng thái xác thực thời gian thực

### Tab 3: Giám sát Webcam
1. Chọn camera để giám sát
2. Nhấn "BẮT ĐẦU GIÁM SÁT"
3. Theo dõi chỉ số rủi ro trực tiếp

### Tab 4: Kiểm thử Video
1. Tải lên tệp video (MP4, MOV)
2. Chờ xử lý phân tích
3. Xem biểu đồ điểm số
4. Đọc báo cáo kết luận

---

## 🔧 Cấu hình Advanced

### Điều chỉnh Ngưỡng Bảo mật

Trong sidebar, sử dụng slider "Ngưỡng bảo mật" để điều chỉnh độ nhạy (0.1 - 0.9):
- **Thấp (0.3)**: Nhạy cảm cao, ít false negative
- **Trung bình (0.57)**: Cân bằng (mặc định)
- **Cao (0.7)**: Giảm false positive

### Tối ưu hóa GPU

Nếu có GPU NVIDIA, hãy cài CUDA version phù hợp:

```bash
# Cho NVIDIA GPU (ví dụ CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔬 Các Thuật toán Chính

### 1. Face Detection & Preprocessing
- **MTCNN**: Phát hiện khuôn mặt đa bước
- **Haar Cascade**: Fallback phát hiện
- **Contextual Padding**: Mở rộng 15% để bắt artifact blending

### 2. Feature Extraction
- **EfficientNet-B4**: Trích xuất đặc trưng từ ImageNet pretrained
- **Frequency Domain Analysis**: FFT để phát hiện artifacts tần số

### 3. Temporal Analysis
- **Sliding Window (k=10)**: Làm mịn dự đoán
- **Identity Drift**: Cosine similarity giữa embeddings liên tiếp
- **Motion Jitter**: Standard deviation của centroid

### 4. Fusion Strategy
```python
risk_score = max(AI_prediction, forensic_score) + uncertainty_penalty
final_verdict = risk_score > threshold
```

---

## 📊 Mô tả Output

### Video Lab Report
```
✅ VIDEO TIN CẬY (NGƯỜI THẬT)

- Điểm trung bình: 0.2345
- Độ phân tán (Std): 0.0856
- Chỉ số P95: 0.3456
- Tỷ lệ frame mạo danh: 2.3%
- Spike bất thường: 0

📊 Hành vi: Ổn định - Giống người thật
```

### eKYC Status
```
🟢 AN TOÀN
Định danh: Đã xác minh
Rủi ro tổng hợp: 18.5%
Identity Drift: 0.123
Motion Jitter: 5.2
```

---

## ⚠️ Giới hạn & Hạn chế

1. **Dependency on Face**: Hệ thống yêu cầu mặt nhìn rõ ràng
2. **Lighting**: Điều kiện ánh sáng tốt sẽ cải thiện độ chính xác
3. **Resolution**: Video quality ≥ 720p khuyên cáo
4. **Model Size**: Weights ~500MB, không phù hợp cho edge device nhẹ

---

## 🤝 Đóng góp & Hợp tác

Chúng tôi chào đón các đóng góp! Vui lòng:

1. Fork repository
2. Tạo branch cho feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📜 License

Dự án này được cấp phép dưới **MIT License** - xem file [LICENSE](LICENSE) để chi tiết.

---

## 📚 Tham khảo

- Tan et al. (2020): "FaceForensics++: Learning to Detect Manipulated Facial Images"
- Li et al. (2021): "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics"
- Tan & Le (2019): "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

---

**⭐ Nếu thấy dự án hữu ích, vui lòng cho một star!**

*Note: Dự án này được thực hiện phục vụ mục đích Nghiên cứu Khoa học. Model weights được lưu trữ ngoài do giới hạn kích thước. Liên hệ tác giả để nhận access.*
