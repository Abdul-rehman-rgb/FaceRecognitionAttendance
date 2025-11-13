import pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# ------------------- Text-to-Speech Function -------------------
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# ------------------- Mobile Camera Setup -------------------
url = "http://192.168.21.119:8080/video"  # Replace with your IP webcam URL
cap = cv2.VideoCapture(url)

# ------------------- Face Detection -------------------
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------- Load Dataset -------------------
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

# ------------------- Train KNN -------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# ------------------- Attendance Folder -------------------
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# ------------------- Optional Background -------------------
# img_bg = cv2.imread('background.jpg')  # Uncomment if using a background
# y_offset, x_offset = 162, 55  # Position to place camera feed on background

# ------------------- Main Loop -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop & resize face
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        face_flatten = resized_img.reshape(1, -1)

        # Predict name
        output = knn.predict(face_flatten)[0]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y-25), (x+w, y), (0, 255, 0), -1)
        cv2.putText(frame, str(output), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ------------------- Attendance Logging -------------------
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        filename = f"Attendance/Attendance_{date}.csv"

        file_exists = os.path.isfile(filename)
        existing_names = []

        if file_exists:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                existing_names = [row[0] for row in reader]

        if output not in existing_names:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['NAME', 'TIME'])
                writer.writerow([output, timestamp])

            # Text-to-Speech announcement
            speak(f"{output} is present")

    # ------------------- Optional Background Overlay -------------------
    # Uncomment below if using a background image
    # cam_h, cam_w, _ = frame.shape
    # display_img = img_bg.copy()
    # display_img[y_offset:y_offset+cam_h, x_offset:x_offset+cam_w] = frame
    # cv2.imshow("Face Recognition", display_img)

    # ------------------- Show Video Feed -------------------
    cv2.imshow("Face Recognition", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

# ------------------- Release Resources -------------------
cap.release()
cv2.destroyAllWindows()
