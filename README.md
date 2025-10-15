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


