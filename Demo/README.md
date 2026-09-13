# Hệ thống Phát hiện Deepfake (Deepfake Detection System)

Dự án Nghiên cứu Khoa học (NCKH) về phát hiện Deepfake sử dụng kiến trúc **EfficientNet-B4** kết hợp với **Học chuyển giao (Transfer Learning)** và các kỹ thuật phân tích thị giác máy tính chuyên sâu.

## 🌟 Tính năng chính

Hệ thống được tích hợp giao diện Web trực quan thông qua Streamlit với 4 phân hệ chính:
1.  **Đối chứng Thực nghiệm (Adversarial Simulator):** So sánh song song luồng Webcam trực tiếp với các video tấn công (Attack Payload) để kiểm tra độ tin cậy.
2.  **Xác thực Sinh trắc (eKYC):** Mô phỏng quy trình định danh khách hàng trực tuyến, tích hợp các thuật toán chống giả mạo (Anti-spoofing).
3.  **Giám sát Webcam:** Theo dõi và cảnh báo rủi ro Deepfake theo thời gian thực từ camera.
4.  **Kiểm thử Video (Video Lab):** Phân tích các tệp video ngoại tuyến, đưa ra báo cáo chi tiết về các chỉ số thống kê (P95, Mean, Spikes, Identity Drift).

## 🛠️ Công nghệ sử dụng

-   **Ngôn ngữ:** Python 3.9+
-   **Deep Learning Framework:** PyTorch
-   **Kiến trúc Model:** EfficientNet-B4 (được tinh chỉnh cho dữ liệu Deepfake)
-   **Xử lý hình ảnh:** OpenCV, MediaPipe, MTCNN
-   **Giao diện:** Streamlit
-   **Phân tích thống kê:** NumPy, SciPy, Pandas

## 🚀 Hướng dẫn cài đặt và chạy

### 1. Cài đặt môi trường

Khuyến nghị sử dụng môi trường ảo (Virtual Environment):

```bash
# Tạo môi trường ảo
python -m venv deepfake_env

# Kích hoạt môi trường ảo (Windows)
.\deepfake_env\Scripts\activate

# Kích hoạt môi trường ảo (Linux/Mac)
source deepfake_env/bin/activate
```

### 2. Cài đặt thư viện

Cài đặt các thư viện cần thiết từ file `requirement.txt`:

```bash
pip install -r requirement.txt
```

*Lưu ý: Nếu bạn có GPU NVIDIA, hãy cài đặt phiên bản PyTorch hỗ trợ CUDA để hệ thống chạy mượt mà hơn.*

### 3. Chuẩn bị Model

Đảm bảo file model `best_pytorch_model_final.pth` đã được đặt trong thư mục `models/`.

### 4. Khởi chạy ứng dụng

Sử dụng lệnh sau để chạy giao diện Web:

```bash
streamlit run video_app.py
```

Sau khi chạy lệnh, trình duyệt sẽ tự động mở trang dashboard (thường là `http://localhost:8501`).

## 📁 Cấu trúc thư mục quan trọng

-   `video_app.py`: Tệp tin chính điều khiển giao diện và logic ứng dụng.
-   `model_pytorch.py`: Định nghĩa kiến trúc mạng EfficientNet-B4.
-   `models/`: Chứa các trọng số (weights) của mô hình đã huấn luyện.
-   `requirement.txt`: Danh sách các thư viện cần cài đặt.
-   `hacker_mask.png`: Tài nguyên dùng cho tính năng mô phỏng tấn công.

## 📊 Chỉ số phân tích

Hệ thống đánh giá dựa trên sự kết hợp của:
-   **AI Prediction Score:** Điểm số từ mô hình Deep Learning.
-   **Forensic Analysis:** Phân tích tần số (FFT) và độ sắc nét (Laplacian) của ảnh.
-   **Identity Drift:** Theo dõi sự thay đổi định danh giữa các frame.
-   **Motion Jitter:** Phân tích sự rung lắc bất thường của vùng mặt.

---
*Dự án được thực hiện phục vụ mục đích Nghiên cứu Khoa học.*
