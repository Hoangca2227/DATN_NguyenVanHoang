import tensorflow as tf
from keras._tf_keras.keras.models import load_model  # Dùng để tải mô hình đã được huấn luyện trước đó.
from time import sleep                               #Tạo độ trễ nếu cần.
from keras_preprocessing.image import img_to_array   #Chuyển đổi ảnh sang dạng mảng để đưa vào mô hình dự đoán.
import cv2                                           # OpenCV để xử lý ảnh và nhận diện khuôn mặt.
import numpy as np
from PIL import ImageFont, ImageDraw, Image          #Hiển thị chữ tiếng Việt hoặc chữ có font tùy chỉnh.

# Tải bộ phân loại khuôn mặt và mô hình
# Nếu chạy trên máy khác thì phải thay đổi đường dẫn
face_classifier = cv2.CascadeClassifier(r"C:\Users\Asus\BTL_TTDN\Emotion_detection\haarcascade_frontalface_default.xml")
classifier = load_model(r"C:\Users\Asus\BTL_TTDN\Emotion_detection\model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Cài đặt fone chữ hiển thị
font_path = "arial.ttf" # Font chữ được sử dụng để hiển thị thông báo trên màn hình (cần có file font này).
font = ImageFont.truetype(font_path, 32) #

cap = cv2.VideoCapture(0) # Bật camera
while True:
    _, frame = cap.read() # Đọc từng khung hình từ camera.
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Chuyển đổi ảnh màu thành ảnh xám.
    faces = face_classifier.detectMultiScale(gray) # Phát hiện khuôn mặt trong ảnh
    # Kiểm tra xem có khuôn mặt hay không
    if len(faces) == 0:  # Không phát hiện thấy khuôn mặt nào
        # Tạo hình ảnh PIL từ khung hình
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        text = "Không phát hiện được khuôn mặt"

        # Lấy kích thước của hộp văn bản
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Đặt chữ ở giữa màn hình
        text_x = (frame.shape[1] - text_width) // 2  # Căn giữa theo chiều ngang
        text_y = (frame.shape[0] - text_height) // 2  # Căn giữa theo chiều dọc

        # Vẽ chữ lên hình
        draw.text((text_x, text_y), text, font=font, fill=(0, 255, 0))

        # Chuyển đổi lại thành định dạng OpenCV
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # Vẽ hình chữ nhật quanh khuôn mặt bằng cv2.rectangle().
            roi_gray = gray[y:y + h, x:x + w] # Cắt vùng khuôn mặt từ ảnh gốc.
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA) # Resize về kích thước 48x48 để phù hợp với đầu vào của mô hình.
            # Chuẩn hóa dữ liệu về khoảng [0,1]
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
            # Chuyển đổi thành mảng NumPy và reshape để phù hợp với mô hình.
                prediction = classifier.predict(roi)[0] # Dự đoán cảm xúc bằng mô hình classifier.predict(roi).
                label = emotion_labels[prediction.argmax()] # Lấy nhãn cảm xúc có xác suất cao nhất.
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Hiển thị nhãn lên khung hình bằng cv2.putText()
            else:
                cv2.putText(frame, 'Không có khuôn mặt nào', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame) # Hiển thị khung hình có khuôn mặt và nhãn cảm xúc.
    if cv2.waitKey(1) & 0xFF == ord('q'): # nhấn q để thoát
        break

cap.release()
cv2.destroyAllWindows()
