import cv2
import pickle
import numpy as np
import os

name = input("Enter Your Name: ")

# Replace this with your phone's IP webcam URL
url = "http://192.168.21.119:8080/video"

# Open the video stream
cap = cv2.VideoCapture(url)

# Haar Cascade for face detection
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

faces_data = []
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w]
        resized_image = cv2.resize(crop_image, (50, 50))

        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_image)

        i += 1

        cv2.putText(frame, str(len(faces_data)), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Mobile Camera - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Collected {len(faces_data)} face samples")

# Convert and reshape
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

# Load existing data if available
if 'names.pkl' not in os.listdir('data'):
    names = [name] * len(faces_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    # Append to existing dataset
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    with open('data/faces.pkl', 'rb') as f:
        existing_faces = pickle.load(f)

    names += [name] * len(faces_data)
    faces_data = np.append(existing_faces, faces_data, axis=0)

    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces.pkl', 'wb') as f:
        pickle.dump(faces_data, f)

print("✅ Data saved successfully!")
