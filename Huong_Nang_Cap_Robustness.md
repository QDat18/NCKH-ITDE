# HƯỚNG NÂNG CẤP HỆ THỐNG (QUAN TRỌNG NHẤT)

Dựa trên kết quả từ bài kiểm tra độ bền vững (`stress_test_robustness.py`), hệ thống hiện tại đang gặp "điểm yếu chí mạng" đối với **Nhiễu cảm biến (Sensor Noise)** khi chỉ số AUC rớt thê thảm xuống `0.5160` (tương đương đoán mò). Đồng thời, hiệu suất cũng suy giảm đáng kể với **Ảnh bị nén (High Compression - AUC 0.8505)** và **Mờ chuyển động (Gaussian Blur - AUC 0.8244)**. Mặc dù hệ thống vượt qua xuất sắc môi trường Low Light.

Để khắc phục triệt để các rào cản trên và nâng cao tính Robustness (Độ bền vững) theo chuẩn Enterprise, dưới đây là **5 hướng nâng cấp cốt lõi**:

### 🔧 1. Data Augmentation mạnh hơn (Bắt buộc)
Mô hình hiện tại đang bị "Overfitting" với các vùng dữ liệu sạch. Việc đầu tiên cần giải quyết trong `dataset_pytorch.py` là tạo ra các luồng dữ liệu nhiễu nhân tạo (Augmentation Pipeline) để "tiêm chủng" cho mạng nơ-ron:
- **Gaussian noise training**: Cố tình chèn độ nhiễu loạn Gauss mạnh vào ~30% - 40% batch ảnh được đưa vào huấn luyện mô hình.
- **JPEG compression augmentation**: Tái tạo hiệu ứng mất mát do nén video trên đường truyền mạng hoặc tải qua các nền tảng MXH (Giảm Quality [20, 50]).
- **Motion blur augmentation**: Áp dụng các thuật toán mờ cục bộ bắt chước hiệu ứng rung lắc của webcam khi người dùng đang di chuyển.
- **Low-light simulation**: Thay đổi độ tương phản, mức phơi sáng và các đốm nhiễu li ti giả lập camera máy tính xách tay giá rẻ.

### 🧠 2. Multi-domain Training (Huấn luyện Đa Miền)
Không chỉ huấn luyện mô hình phân biệt REAL/FAKE trên "miền ảnh gốc" (Clean Domain), chúng ta sẽ huấn luyện đa nhiệm để mô hình phân loại song song:
- **Train thêm với nhóm Noisy Frames (Miền Tín hiệu Nhiễu)**: Đưa các clip bị đánh nhiễu từ đầu vào giúp mô hình học cách phân biệt sự khác nhau giữa "nhiễu vật lý do môi trường" so với "artifacts do AI sinh ra".
- **Compressed video frames (Miền Nén)**: Học các đặc trưng Deepfake đã đi qua bộ lọc nén của WhatsApp, Zalo hay Messenger.

### 🔬 3. Frequency-aware Model (Rất quan trọng)
Sự sụp đổ của AUC tại điểm nhiễu Gauss chứng tỏ EfficientNet-B0 cực kì dễ bị đánh lừa trong miền Không gian thời gian thực (RGB Domain). Việc nâng cấp sang miền Tần số (Frequency Domain) là lối thoát hiểm tốt nhất:
- **FFT Branch CNN**: Xây dựng một mạng luồng đôi (Two-stream Network). Luồng 1 tiếp nhận ảnh RGB. Luồng 2 tiếp nhận ảnh đã qua biến đổi **Fast Fourier Transform (FFT)**. Các artifacts giả mạo thường để lộ rõ cực kỳ trong phổ biên độ miền tần số.
- **High/Low frequency split learning**: Tách biệt vùng học phần tần số thấp (kết cấu da tự nhiên) với vùng học phần tần số cao (lỗi biên viền, noise giả mạo ảnh).

### 🧩 4. Temporal Smoothing (Giảm Noise Impact)
Một frame đơn lẻ có nhiễu quá nặng thể làm hỏng dự đoán chung. Cần bổ sung kết nối không gian-thời gian:
- **Frame aggregation**: Tổng hợp đặc trưng (Feature Map Aggregation) trích xuất từ mô hình CNN của chuỗi N frame liên tiếp tại các lớp LSTM/GRU để tránh việc mô hình đánh giá rời rạc.
- **Temporal voting**: Cơ chế "Cửa sổ dự đoán k=10" hiện tại phải thay thế bình quân bằng **Tham số tính toán trọng số (Weighted Voting)**. Các khung hình bị hệ thống xác định có độ mù và độ nén cao sẽ tự động bị giảm trọng số ra quyết định.

### 🛡 5. Robust Fusion Update
Hệ thống kết hợp phân tích Forensic và phân tích Nhận diện Sinh trắc đang bị Noise kéo tụt điểm tổng quát. Do đó kiến trúc ra quyết định cuối cùng cần có "lớp đề kháng bổ sung":
- **Công thức Cập nhật**: Căn chỉnh lại hàm trích điểm số:
  `final_score = weighted_average + uncertainty_penalty`
- **Cơ chế bẻ lái**: Khi nhận thấy một frame có mức độ nhiễu (Noise) quá cao vuợt ngưỡng đọc thông tin, mô hình sẽ nâng hệ số *uncertainty_penalty*. Khi sự không chắc chắn lên đỉnh điểm, mô hình từ chối kết luận FAKE/REAL và kích hoạt UI/UX yêu cầu: "Chất lượng đường truyền kém hoặc ảnh quá mờ, vui lòng thử lại." Thay vì cố ép kết quả dẫn đến False Positives tồi tệ.
