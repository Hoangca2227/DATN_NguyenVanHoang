# 🎭 Phát triển hệ thống nhận diện và phân tích biểu cảm khuôn mặt dựa trên mô hình FaceNet

## 🧠 Giới thiệu
Đề tài tập trung xây dựng **hệ thống nhận diện khuôn mặt và phân tích biểu cảm cảm xúc** của con người trong thời gian thực.  
Hệ thống kết hợp nhiều mô hình học sâu để xử lý các bước khác nhau trong pipeline nhận diện khuôn mặt:

- **MTCNN (Multi-task Cascaded Convolutional Networks)** – phát hiện và cắt khuôn mặt từ ảnh/video.
- **FaceNet** – trích xuất đặc trưng (embedding) 512 chiều đại diện cho từng khuôn mặt.
- **SVM (Support Vector Machine)** – phân loại danh tính khuôn mặt dựa trên vector đặc trưng từ FaceNet.
- **CNN (Convolutional Neural Network)** – phân tích và xác định **biểu cảm khuôn mặt** (ví dụ: vui, buồn, giận, ngạc nhiên, sợ hãi, ghê tởm, bình thường).

---

## ⚙️ Kiến trúc hệ thống

```mermaid
graph TD
A[Input: Ảnh/Video] --> B[MTCNN: Phát hiện khuôn mặt]
B --> C[FaceNet: Trích xuất đặc trưng khuôn mặt]
C --> D1[SVM: Nhận diện danh tính]
C --> D2[CNN: Phân tích biểu cảm]
D1 --> E[Hiển thị tên và ID khuôn mặt]
D2 --> F[Hiển thị biểu cảm tương ứng]

| Thành phần           | Mô hình / Thư viện            | Mô tả                                                    |
| -------------------- | ----------------------------- | -------------------------------------------------------- |
| Phát hiện khuôn mặt  | **MTCNN** (`mtcnn`)           | Xác định vị trí khuôn mặt và landmark (mắt, mũi, miệng). |
| Trích xuất đặc trưng | **FaceNet** (`keras-facenet`) | Tạo vector đặc trưng 512 chiều cho mỗi khuôn mặt.        |
| Nhận diện danh tính  | **SVM** (`scikit-learn`)      | Phân loại khuôn mặt dựa trên vector embedding.           |
| Phân tích biểu cảm   | **CNN** (`TensorFlow/Keras`)  | Dự đoán biểu cảm từ ảnh khuôn mặt đã chuẩn hóa.          |
| Giao diện demo       | **OpenCV**                    | Hiển thị kết quả theo thời gian thực.                    |

🚀 Cách chạy chương trình
1️⃣ Cài đặt môi trường
pip install -r requirements.txt

2️⃣ Huấn luyện hoặc tải mô hình sẵn có


Chạy facenet.ipynb để huấn luyện mô hình nhận diện khuôn mặt.

Chạy emotion_detion.ipynb để huấn luyện mô hình phân tích biểu cảm.

3️⃣ Chạy demo nhận diện & phân tích biểu cảm
python main.py


👉 Hệ thống sẽ bật webcam và hiển thị:

Tên người dùng nhận diện được.

Biểu cảm khuôn mặt dự đoán theo thời gian thực.

🧪 Kết quả đạt được

Độ chính xác nhận diện khuôn mặt (FaceNet + SVM): ~98% trên tập thử nghiệm.

Độ chính xác phân tích biểu cảm (CNN): ~90% trên tập dữ liệu FER2013.

Chạy ổn định real-time (≈25 FPS) với webcam laptop.

💡 Hướng phát triển

Tối ưu tốc độ xử lý bằng TensorRT hoặc ONNX Runtime.

Tích hợp với hệ thống điểm danh, giám sát an ninh, hoặc chatbot cảm xúc.

Phát triển giao diện web với Django hoặc Streamlit.
