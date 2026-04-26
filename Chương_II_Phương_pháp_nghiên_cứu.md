# CHƯƠNG II: PHƯƠNG PHÁP NGHIÊN CỨU

## 2.1. Phát biểu bài toán
Trong bối cảnh công nghệ trí tuệ nhân tạo (AI) phát triển bùng nổ, kỹ thuật Deepfake đã trở thành một thách thức lớn đối với an ninh thông tin và sự tin cậy trong giao tiếp số. Bài toán nghiên cứu tập trung vào việc **nhận diện các khuôn mặt bị chỉnh sửa hoặc giả mạo bằng AI (Deepfake Detection)** trong các luồng dữ liệu video. 

Mục tiêu cụ thể là xây dựng một hệ thống phân loại nhị phân (Binary Classification) có khả năng phân biệt giữa khuôn mặt thực (Real) và khuôn mặt giả lập (Fake). Bài toán yêu cầu mô hình không chỉ đạt độ chính xác cao mà còn phải đảm bảo khả năng **vận hành theo thời gian thực (real-time)**, phù hợp để tích hợp vào các ứng dụng gọi video (Video Call) hoặc giám sát trực tuyến. Các thách thức chính bao gồm: tính đa dạng của các kỹ thuật tạo Deepfake (với độ chân thực ngày càng cao), sự biến đổi về điều kiện ánh sáng, góc quay và yêu cầu khắt khe về độ trễ (latency).

## 2.2. Kiến trúc mô hình EfficientNet và Kỹ thuật Học chuyển giao
### 2.2.1. Kiến trúc EfficientNet
Để giải quyết bài toán cân bằng giữa hiệu năng và chi phí tính toán, nghiên cứu đề xuất sử dụng kiến trúc **EfficientNet-B0**. Điểm ưu việt của EfficientNet nằm ở phương pháp **Compound Scaling** (tổng hợp tỷ lệ), thực hiện việc mở rộng mạng một cách đồng bộ theo ba chiều: độ sâu (depth), độ rộng (width) và độ phân giải của ảnh đầu vào (resolution).

Thay vì chỉ mở rộng một chiều một cách tùy ý như các kiến trúc truyền thống, EfficientNet sử dụng hệ số tỉ lệ đồng nhất $\phi$ để tối ưu hóa nguồn lực. Điều này giúp EfficientNet-B0 có kích thước nhỏ gọn hơn đáng kể (số lượng tham số ít hơn) so với các mạng như ResNet hay VGG nhưng vẫn đạt được độ chính xác tương đương hoặc vượt trội, rất phù hợp cho các bài toán nhận diện trực tiếp trên thiết bị có nguồn lực giới hạn.

### 2.2.2. Kỹ thuật Học chuyển giao (Transfer Learning)
Do việc thu thập và huấn luyện một mạng nơ-ron sâu từ đầu (from scratch) đòi hỏi tập dữ liệu khổng lồ và tài nguyên tính toán lớn, kỹ thuật **Học chuyển giao** đã được áp dụng. Nghiên cứu sử dụng các trọng số (weights) đã được huấn luyện sẵn trên tập dữ liệu **ImageNet** – một tập dữ liệu quy mô lớn với hàng triệu hình ảnh thuộc 1000 lớp khác nhau.

Thông qua việc kế thừa khả năng trích xuất đặc trưng (feature extraction) mạnh mẽ từ ImageNet, mô hình chỉ cần trải qua quá trình tinh chỉnh (Fine-tuning) trên tập dữ liệu Deepfake. Phương pháp này giúp mô hình hội tụ nhanh hơn và đạt được độ chính xác cao ngay cả khi tập dữ liệu huấn luyện cụ thể không quá lớn.

## 2.3. Mô tả và Tiền xử lý dữ liệu

### 2.3.1. Tập dữ liệu và Chiến lược cân bằng mẫu (Sampling Strategy)
Nghiên cứu sử dụng tập dữ liệu **Celeb-DF v2** làm nguồn dữ liệu chính. Đây là bộ dữ liệu thực tế bao gồm 818 video gốc (Real) và 5.639 video giả mạo (Fake). Một thách thức lớn của tập dữ liệu này là sự mất cân bằng lớp nghiêm trọng (tỷ lệ xấp xỉ 1:7). 

Để giải quyết vấn đề này và đảm bảo mô hình không bị thiên kiến (bias) về phía lớp chiếm ưu thế, nghiên cứu áp dụng chiến lược **Cân bằng mẫu (Under-sampling)**:
- **Lớp Real:** Giữ nguyên toàn bộ 818 mẫu video gốc.
- **Lớp Fake:** Thực hiện lấy mẫu ngẫu nhiên (Random Sampling) một lượng dữ liệu từ 5.639 video giả mạo để đạt được tỷ lệ cân bằng 1:1 với lớp Real.
Chiến lược này giúp hàm mất mát (loss function) hội tụ một cách khách quan nhất, tránh trường hợp mô hình chỉ tối ưu cho lớp có nhiều dữ liệu hơn.

### 2.3.2. Phân chia tập dữ liệu (Data Splitting)
Sau khi thực hiện cân bằng và trích xuất khung hình, tập dữ liệu tổng hợp được xáo trộn ngẫu nhiên (shuffle) và phân chia theo tỷ lệ **80:10:10**, cụ thể:
- **Tập Huấn luyện (Train set - 80%):** Dùng để cập nhật trọng số của mạng nơ-ron thông qua quá trình lan truyền ngược (backpropagation).
- **Tập Kiểm thử (Validation set - 10%):** Dùng để đánh giá hiệu suất mô hình sau mỗi epoch, điều chỉnh siêu tham số và thực hiện cơ chế dừng sớm (Early Stopping).
- **Tập Đánh giá (Test set - 10%):** Dùng để đưa ra các chỉ số khách quan cuối cùng (Accuracy, Precision, Recall, F1-Score) sau khi quá trình huấn luyện kết thúc hoàn toàn.

### 2.3.3. Quy trình tiền xử lý kỹ thuật (Preprocessing Pipeline)
Dữ liệu video thô được đưa qua quy trình xử lý tự động để trích xuất các đặc trưng hình thái khuôn mặt có giá trị:

**1. Định vị và Canh lề sinh trắc học (Face Detection & Alignment):**
Để loại bỏ sự nhiễu loạn từ hậu cảnh (background noise), nghiên cứu tích hợp mạng nơ-ron phân tầng **MTCNN** (Multi-task Cascaded Convolutional Networks) làm bộ dò tìm khuôn mặt. Thuật toán này vận hành qua ba lớp mạng liên hoàn (P-Net, R-Net, O-Net) để dự đoán tọa độ hộp giới hạn (bounding box) và tinh chỉnh vị trí 5 điểm mốc sinh trắc học cơ sở (mắt, mũi, khóe miệng). Sự ưu việt của MTCNN nằm ở khả năng khoanh vùng chính xác khuôn mặt ngay cả khi đối tượng có sự biến thiên phức tạp về góc xoay đầu (pose variations) hay bị che khuất một phần (partial occlusion) trong điều kiện thực tế.

```python
from mtcnn import MTCNN
import cv2

# Khởi tạo bộ dò tìm MTCNN
detector = MTCNN()

def extract_face(frame):
    # Chuyển đổi màu sang RGB cho MTCNN
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)
    
    if results:
        # Chọn khuôn mặt có độ tin cậy cao nhất
        face = max(results, key=lambda x: x["confidence"])
        x, y, w, h = face["box"]
        return x, y, w, h
    return None
```

**2. Kỹ thuật Mở rộng không gian biên (Contextual Margin Padding):**
Đây là một bước can thiệp mang tính bước ngoặt đối với bài toán nhận diện Deepfake. Khác với các bài toán phân loại khuôn mặt thông thường, đặc trưng của kỹ thuật tạo ảnh mạo danh bằng AI (như Autoencoder hay GAN) thường để lại các dấu vết khiếm khuyết nội suy (blending artifacts) tại khu vực ranh giới ghép nối giữa khuôn mặt giả và hậu cảnh thật.

Do đó, trước khi tiến hành cắt ảnh (cropping), thuật toán chủ động mở rộng hộp giới hạn thêm một khoảng đệm an toàn bằng 15% diện tích. Thao tác toán học này cung cấp cho các lớp tích chập của mạng EfficientNet một tầm nhìn bao quát hơn về vùng không gian chuyển tiếp (transition boundaries) xung quanh viền hàm, cổ và chân tóc – nơi các đứt gãy về phổ màu và độ sắc nét thường bộc lộ rõ nhất.

```python
def apply_padding(x, y, w, h, frame_shape):
    # Tính toán khoảng đệm 15% chiều rộng khuôn mặt
    pad = int(0.15 * w)
    
    # Xác định tọa độ mới đảm bảo không vượt quá biên ảnh gốc
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame_shape[1], x + w + pad), min(frame_shape[0], y + h + pad)
    
    return x1, y1, x2, y2

# Thực thi trích xuất vùng đặc trưng (Region of Interest)
x1, y1, x2, y2 = apply_padding(x, y, w, h, frame.shape)
face_context = frame[y1:y2, x1:x2]
```

**3. Lọc nhiễu mờ động bằng Toán tử không gian (Mathematical Blur Filtering):**
Các khung hình bị nhòe mờ do chuyển động nhanh của đối tượng (motion blur) sẽ làm mất đi các chi tiết tần số cao (đường viền, góc cạnh), dẫn đến việc mạng nơ-ron cập nhật gradient sai lệch (noisy gradients). Để kiểm soát rủi ro này, nghiên cứu áp dụng Toán tử Laplacian bậc hai để đo lường độ sắc nét của ma trận điểm ảnh. 

Bản chất toán học của Laplacian là tính toán đạo hàm bậc hai của ảnh để phát hiện các cạnh sắc nét. Thuật toán sẽ tính toán phương sai (variance) cường độ sáng của từng khung hình; hình ảnh càng sắc nét, phương sai càng lớn. Bằng thực nghiệm, ngưỡng đào thải được thiết lập tại $\tau = 80$. Mọi khung hình có chỉ số phương sai rơi xuống dưới ngưỡng này sẽ tự động bị phân loại là ảnh nhiễu và bị loại bỏ hoàn toàn khỏi không gian mẫu huấn luyện.

```python
def is_blurry(image, threshold=80):
    # Chuyển đổi sang ảnh xám để tính toán đạo hàm
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tính toán bộ lọc Laplacian và lấy phương sai
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Kiểm tra theo ngưỡng đào thải thực nghiệm
    return laplacian_var < threshold

# Triển khai trong quy trình lọc dữ liệu sạch
if is_blurry(face_crop):
    print("Mẫu bị loại bỏ do nhiễu mờ động.")
    continue
```

**4. Nội suy và Chuẩn hóa tham số (Normalization & Resizing):**
Tại bước cuối cùng, mẫu hình khuôn mặt được nội suy về độ phân giải tiêu chuẩn $224 \times 224$ pixel nhằm tương thích tuyệt đối với cấu trúc đầu vào của kiến trúc EfficientNet. Tiếp theo, hệ thống thực hiện phép Chuẩn hóa Z-score (Z-score normalization) trên ma trận ba kênh màu (RGB) dựa trên các hệ số kỳ vọng (mean) và độ lệch chuẩn (std) của tập dữ liệu ImageNet. Phép biến đổi này giúp dịch chuyển phân phối dữ liệu về điểm trung tâm, thu hẹp không gian tìm kiếm bề mặt hàm mất mát, từ đó gia tăng đáng kể tốc độ và độ ổn định khi hội tụ của thuật toán tối ưu.

```python
from torchvision import transforms

# Quy trình chuyển đổi và chuẩn hóa dữ liệu
preprocess = transforms.Compose([
    # Nội suy về độ phân giải tiêu chuẩn cho EfficientNet-B0
    transforms.Resize((224, 224)),
    
    # Chuyển đổi sang Tensor (Scale về [0, 1])
    transforms.ToTensor(),
    
    # Chuẩn hóa Z-score theo thông số ImageNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```

## 2.4. Xây dựng và Huấn luyện mô hình đề xuất

Sau khi hoàn tất các bước tiền xử lý chuyên sâu cho dữ liệu hình ảnh, nhóm tiến hành xây dựng mô hình học sâu (Deep Learning) tối ưu nhằm nhận diện các đặc trưng giả mạo. Mô hình được triển khai trên nền tảng thư viện **PyTorch** với kiến trúc cốt lõi dựa trên **EfficientNet-B0**. Cụ thể, mô hình khai thác sức mạnh của các khối nơ-ron tích chập mở rộng (MBConv) phối hợp với cơ chế *Squeeze-and-Excitation (SE)* để tự động tái trọng số các kênh đặc trưng quan trọng. Kiến trúc này không những giúp mô hình nắm bắt được các chi tiết tinh vi về kết cấu da và ranh giới điểm ảnh (pixel boundaries) mà còn tối ưu hóa tài nguyên tính toán, đảm bảo khả năng nhận diện chính xác và ổn định trên các luồng video thời gian thực.

### 2.4.1. Kiến trúc mạng phân loại (Classification Head)
Mô hình sử dụng mạng xương sống (backbone) là **EfficientNet-B0** đã được huấn luyện sẵn trọng số trên tập ImageNet. Toàn bộ lớp phân loại gốc được thay thế bằng một cấu trúc tùy chỉnh phục vụ bài toán phân loại nhị phân thực tế:

```python
import torch.nn as nn
from timm import create_model

class DeepfakeEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load backbone B0
        self.backbone = create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # Đưa vào Classification Head tùy chỉnh
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512), # 1280 là đặc trưng đầu ra của B0
            nn.ReLU(),
            nn.Dropout(0.3),       # Chống overfitting
            nn.Linear(512, 1)      # Output Logit (0 hoặc 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

### 2.4.2. Chiến lược huấn luyện và Thuật toán tối ưu
Để mô hình hội tụ nhanh và ổn định, nghiên cứu áp dụng các cấu hình tối ưu sau:
- **Hàm mất mát (Loss Function):** Sử dụng `BCEWithLogitsLoss`. Đây là sự kết hợp giữa lớp Sigmoid và Binary Cross Entropy, giúp tăng cường độ ổn định số học trong quá trình tính toán đạo hàm.
- **Thuật toán tối ưu (Optimizer):** Sử dụng **Adam** với $LR = 10^{-4}$. Adam giúp điều chỉnh tốc độ học một cách thích ứng cho từng tham số dựa trên các mô-men bậc nhất và bậc hai của gradient.
- **Batch Size:** Thiết lập tại 16 mẫu/lô để cân bằng giữa tốc độ tính toán và độ mượt của hàm lỗi.

```python
import torch.nn as nn
import torch.optim as optim

# Thiết lập hàm mất mát và thuật toán tối ưu
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # LR = 10^-4
```

### 2.4.3. Kiểm soát hội tụ và Early Stopping
Nhằm ngăn chặn hiện tượng học thuộc lòng dữ liệu huấn luyện (Overfitting), nghiên cứu tích hợp cơ chế **Early Stopping** với độ kiên nhẫn (patience) là 5 epoch. Hệ thống liên tục giám sát giá trị `val_loss` sau mỗi chu kỳ:

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # Nếu loss không giảm đáng kể
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Nếu loss đạt giá trị mới thấp hơn
            self.best_loss = val_loss
            self.counter = 0

# Tích hợp vào vòng lặp huấn luyện
early_stopping = EarlyStopping(patience=5)
# ... sau khi tính toán val_loss ...
early_stopping(val_loss)
if early_stopping.early_stop:
    print("Early stopping triggered!")
    break
```

## 2.5. Chiến lược Inference thực tế và Cơ chế làm mịn (Smoothing)

Để tích hợp mô hình vào các ứng dụng gọi video thực tế, nghiên cứu phát triển một quy trình Inference được tối ưu hóa nhằm cân bằng giữa độ chính xác và trải nghiệm người dùng:

### 2.5.1. Kỹ thuật Bỏ qua khung hình (Frame Skipping)
Thay vì thực hiện dự đoán trên mọi khung hình nhận được từ Webcam (có thể lên tới 30-60 FPS), hệ thống chỉ thực hiện suy luận trên mỗi $n=3$ khung hình. Kỹ thuật này giúp giảm tải cho CPU/GPU mà vẫn đảm bảo tốc độ phản hồi cần thiết cho bài toán thời gian thực.

### 2.5.2. Cơ chế làm mịn bằng Cửa sổ thời gian (Temporal Window)
Các biến động nhỏ về ánh sáng hoặc tư thế mặt có thể gây ra hiện tượng "flickering" (nhảy kết quả liên tục giữa Real và Fake). Để giải quyết vấn đề này, nghiên cứu sử dụng thuật toán **Cửa sổ trượt (Sliding Window)**:
1. Lưu trữ $k=10$ kết quả dự đoán gần nhất vào một hàng đợi (Queue).
2. Kết quả hiển thị cuối cùng cho người dùng là giá trị trung bình (Average) của $k$ dự đoán này. 
Cơ chế này giúp "làm mịn" các sai số tức thời và tăng độ tin cậy của thông tin cảnh báo đối với người dùng cuối.

## Tiểu kết chương II
Trong chương này, nghiên cứu đã trình bày chi tiết về phương pháp tiếp cận bài toán phát hiện Deepfake thông qua mô hình EfficientNet-B0 và môi trường PyTorch. Từ việc xử lý cân bằng dữ liệu Celeb-DF v2, quy trình tiền xử lý kỹ thuật nghiêm ngặt đến việc xây dựng kiến trúc mạng và chiến lược Inference thời gian thực. Đây là nền tảng quan trọng giúp hệ thống không chỉ đạt độ chính xác cao trong phòng thí nghiệm mà còn vận hành bền bỉ và ổn định trong các tình huống thực tế của ứng dụng Video Call.
