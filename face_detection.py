import cv2
import os

# Load the Haar Cascade model
cascade_path = "/home/hitesh-labh/Attendance_project/venv/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(cascade_path)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Error: Could not access webcam!")
else:
    print("✅ Webcam is working. Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("❌ Error: Failed to capture image!")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
