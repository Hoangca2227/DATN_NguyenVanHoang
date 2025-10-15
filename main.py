# =======================
# FACE RECOGNITION + EMOTION RECOGNITION (ver.2)
# =======================

# IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
import mediapipe as mp

# =======================
# INITIALIZATION
# =======================
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']

# Encode label for face recognition
encoder = LabelEncoder()
encoder.fit(Y)

# Load SVM model (for face recognition)
model_face = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Load Emotion model (for facial expression recognition)
model_emotion = tf.keras.models.load_model("model.h5")

# Define emotion label list (sửa theo số lớp bạn train)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# =======================
# VIDEO CAPTURE
# =======================
cap = cv.VideoCapture(0)  # có thể đổi 1 thành 0 tùy camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Lấy tọa độ khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # Đảm bảo bounding box nằm trong khung
            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)

            face_rgb = rgb_frame[y:y+h, x:x+w]
            if face_rgb.size == 0:
                continue

            # =======================
            # 1️⃣ FACE RECOGNITION
            # =======================
            face_resized = cv.resize(face_rgb, (160, 160))
            face_resized = np.expand_dims(face_resized, axis=0)
            ypred = facenet.embeddings(face_resized)
            face_name_pred = model_face.predict(ypred)
            final_name = encoder.inverse_transform(face_name_pred)[0]

            # =======================
            # 2️⃣ EMOTION RECOGNITION
            # =======================
            face_gray = cv.cvtColor(face_rgb, cv.COLOR_RGB2GRAY)
            face_gray = cv.resize(face_gray, (48, 48))  # kích thước chuẩn của model cảm xúc
            face_gray = face_gray / 255.0
            face_gray = np.expand_dims(face_gray, axis=(0, -1))
            emotion_pred = model_emotion.predict(face_gray, verbose=0)
            emotion_label = emotion_labels[np.argmax(emotion_pred)]

            # =======================
            # DISPLAY RESULTS
            # =======================
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(frame, f'{final_name} | {emotion_label}', (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow("Face & Emotion Recognition", frame)
    if cv.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv.destroyAllWindows()
