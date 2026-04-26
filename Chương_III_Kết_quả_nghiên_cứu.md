# CHƯƠNG III: KẾT QUẢ NGHIÊN CỨU

## 3.1. Kết quả huấn luyện mô hình phát hiện Deepfake
Quá trình huấn luyện mô hình EfficientNet-B0 trên tập dữ liệu tinh chỉnh từ Celeb-DF v2 đã cho thấy sự hội tụ ổn định qua từng kỷ nguyên (epoch). 
- **Sự hội tụ:** Đồ thị hàm mất mát (loss) giảm mạnh trong 10 epoch đầu tiên và ổn định dần ở các epoch tiếp theo, chứng tỏ thuật toán Adam đã tối ưu hóa hiệu quả các trọng số của mạng.
- **Độ chính xác:** Mô hình đạt độ chính xác (Accuracy) trên tập kiểm tra vượt mức 96%. Việc sử dụng kỹ thuật trích xuất khuôn mặt bằng MTCNN kết hợp với Padding 15% giúp mạng nơ-ron tập trung sâu vào các đặc trưng biểu cảm và các dấu vết bất thường ở vùng biên khuôn mặt – nơi thường để lại các "artifacts" của thuật toán tạo Deepfake.

## 3.2. Đánh giá hiệu suất của mô hình
Hiệu suất của mô hình đề xuất được đánh giá dựa trên hai khía cạnh chính: độ chính xác phân loại và tốc độ xử lý thực tế.

### 3.2.1. Ma trận nhầm lẫn và Các chỉ số phân loại
Trong bài toán phát hiện hình ảnh giả mạo (Deepfake), lớp dữ liệu mục tiêu cần được nhận diện với độ chính xác cao nhất là lớp Dương tính (Positive - P), tương ứng với các khuôn mặt hoặc video bị thao túng bởi trí tuệ nhân tạo. Việc nhận diện đúng lớp này mang tính sống còn, bởi việc bỏ lọt một trường hợp giả mạo (Deepfake) trong các hệ thống định danh điện tử (eKYC) hay giao tiếp trực tuyến có thể dẫn đến hậu quả nghiêm trọng về bảo mật thông tin và gian lận tài chính. Ngược lại, lớp Âm tính (Negative - N) đại diện cho các hình ảnh, video nguyên bản của người dùng thực.

Để đánh giá hiệu quả phân loại của mô hình, nghiên cứu sử dụng ma trận nhầm lẫn (Confusion Matrix) để thống kê kết quả dự đoán trên toàn bộ tập kiểm tra:

**Hình 17: Ma trận nhầm lẫn trên tập kiểm tra (Epoch 22)**
![confusion_matrix](file:///d:/KyV_HocVienNganHang/NCKH/Final/Demo/confusion_matrix_epoch_22.png)
*Nguồn: Nhóm nghiên cứu*

Trong đó, bốn loại kết quả được xác định như sau:
- **True Positive (TP):** Mô hình dự đoán đúng khuôn mặt Deepfake.
- **True Negative (TN):** Mô hình dự đoán đúng khuôn mặt Real.
- **False Positive (FP):** Mô hình dự đoán nhầm khuôn mặt Real thành Deepfake (Sai lầm loại I).
- **False Negative (FN):** Mô hình dự đoán nhầm khuôn mặt Deepfake thành Real (Sai lầm loại II).

Dựa trên ma trận, ta thấy mô hình có tỷ lệ **True Positive** và **True Negative** rất cao, minh chứng cho khả năng nhận diện chính xác đồng đều trên cả hai lớp dữ liệu.

### 3.2.2. Hiệu năng thời gian thực (Real-time performance)
Đây là điểm nhấn quan trọng nhất của nghiên cứu. Thử nghiệm trên hệ thống máy tính cá thông thường với webcam:
- **Độ trễ (Latency):** Thời gian suy luận (inference) cho mỗi khung hình đạt trung bình từ **30ms đến 50ms**.
- **Tốc độ khung hình (FPS):** Hệ thống duy trì ổn định ở mức **25-30 FPS**, đảm bảo trải nghiệm video mượt mà không bị giật lag.
- **Độ ổn định:** Nhờ vào cơ chế **Temporal Window** (Cửa sổ thời gian $k=10$), các kết quả dự đoán "Real" hoặc "Fake" không bị nhảy (flicker) liên tục khi người dùng thay đổi tư thế, giúp tăng đáng kể trải nghiệm người dùng cuối.

## 3.3. Ưu điểm và hạn chế của mô hình đề xuất

Dựa trên quá trình thực nghiệm và đánh giá thực tế, nghiên cứu xác định các đặc điểm cốt lõi của hệ thống như sau:

### 3.3.1. Ưu điểm
1.  **Tính tối ưu về tài nguyên:** Sử dụng kiến trúc **EfficientNet-B0** giúp cân bằng hoàn hảo giữa độ chính xác và chi phí tính toán. Với số lượng tham số ít hơn đáng kể so với các dòng ResNet hay VGG, mô hình có khả năng triển khai trên các thiết bị đầu cuối (Edge devices) mà không cần GPU hiệu năng cao.
2.  **Độ tin cậy của dữ liệu huấn luyện:** Quy trình tiền xử lý đa tầng (MTCNN + Laplacian Filter + Padding 15%) đã tạo ra một tập dữ liệu "sạch". Việc loại bỏ các khung hình mờ và giữ lại vùng biên giúp mô hình học được các đặc trưng "tĩnh" (texture) và "động" (artifacts) đặc thù của Deepfake.
3.  **Tính ổn định vượt trội:** Nhờ cơ chế **Temporal Smoothing** (Làm mịn theo thời gian), hệ thống khắc phục được hiện tượng nhảy nhãn (flickering) khi gặp nhiễu ánh sáng tức thời, mang lại trải nghiệm người dùng tự nhiên và chính xác trong các cuộc gọi video thực tế.
4.  **Khả năng tổng quát hóa (Generalization):** Chỉ số AUC đạt **0.94** minh chứng cho việc mô hình không chỉ "học thuộc" dữ liệu mà thực sự hiểu được sự khác biệt về bản chất giữa da mặt người thật và các lớp phủ nội suy (interpolation) của AI.

### 3.3.2. Hạn chế
1.  **Sự phụ thuộc vào điều kiện ánh sáng:** Trong môi trường ánh sáng quá yếu hoặc ngược sáng mạnh, bộ dò tìm MTCNN có thể gặp khó khăn trong việc định vị 5 điểm mốc sinh trắc học, từ đó làm giảm hiệu quả trích xuất khuôn mặt.
2.  **Độ nhạy với vật cản (Occlusion):** Các mẫu Deepfake có vật cản lớn che khuất một phần mặt (như đeo khẩu trang, dùng tay che miệng) vẫn là thách thức lớn đối với mô hình, do các đặc trưng tại vùng biên bị đứt gãy.
3.  **Tập dữ liệu nguồn:** Nghiên cứu hiện tại tập trung chủ yếu vào tập Celeb-DF v2. Để tăng cường khả năng nhận diện các loại Deepfake mới nhất (như Diffusion-based Deepfakes), cần bổ sung các nguồn dữ liệu đa dạng hơn về đặc điểm nhân chủng học và kỹ thuật sinh ảnh.

## 3.4. Triển khai mô hình vào thực tế
Mô hình đã được triển khai dưới dạng một ứng dụng **Video Call Simulation** hoàn chỉnh với các đặc tính kỹ thuật sau:
- **Tích hợp MediaPipe:** Sử dụng giải pháp MediaPipe Face Detection cho khâu phát hiện khuôn mặt ở giai đoạn Inference giúp tối ưu hóa tài nguyên hệ thống hơn so với MTCNN trong lúc vận hành thực tế.
- **Quy trình xử lý song song:** Việc bỏ qua khung hình (Frame Skipping) và sử dụng hàng đợi (Queue) giúp hệ thống cân đối giữa hiệu suất CPU và độ chính xác dự đoán.
- **Giao diện hiển thị trực quan:** Ứng dụng hiển thị khung bao (Bounding Box) màu xanh cho "REAL" và màu đỏ cho "FAKE", kèm theo xác suất dự đoán và thông tin độ trễ thời gian thực để người dùng dễ dàng theo dõi.
- **Tối ưu hóa TFLite:** Việc sử dụng định dạng TFLite cho phép mô hình dễ dàng mở rộng sang các nền tảng di động hoặc web trong tương lai.

## Tiểu kết chương III
Kết quả nghiên cứu tại Chương III đã minh chứng cho tính hiệu quả của phương pháp đề xuất. Mô hình không chỉ đạt được các chỉ số phân loại ấn tượng trên tập dữ liệu chuẩn Celeb-DF v2 mà còn thể hiện năng lực vận hành thực tế xuất sắc với độ trễ thấp và tính ổn định cao. Những ưu điểm về kiến trúc và kỹ thuật tối ưu hóa giúp hệ thống sẵn sàng cho việc triển khai rộng rãi, dù vẫn còn một số hạn chế nhỏ về điều kiện môi trường cần được cải thiện trong các nghiên cứu tiếp theo.
