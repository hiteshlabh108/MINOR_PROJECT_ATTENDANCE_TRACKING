import cv2
import dlib
import numpy as np
from face_detection import detect_faces  # Import Haar Cascade-based face detection

# Load Dlib's shape predictor
predictor_path = "/home/hitesh-labh/Attendance_project/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def preprocess_face(image):
    """Detects, aligns, crops, and normalizes faces in an image."""
    
    faces = detect_faces(image)  # Get bounding boxes using Haar Cascades

    if len(faces) == 0:
        print("❌ No face detected.")
        return None  

    processed_faces = []

    for (x, y, w, h) in faces:
        # Crop the detected face
        cropped_face = image[y:y+h, x:x+w]

        # Convert to grayscale for Dlib processing
        gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

        # Convert Haar Cascade box into Dlib rectangle format
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Detect facial landmarks using Dlib
        landmarks = predictor(gray, dlib_rect)
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

        # Compute center of both eyes
        eye_center = (left_eye + right_eye) // 2

        # Compute alignment angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate the cropped face
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1)
        aligned_face = cv2.warpAffine(cropped_face, M, (w, h))

        # Resize and normalize
        resized_face = cv2.resize(aligned_face, (128, 128))
        normalized_face = resized_face.astype("float32") / 255.0

        processed_faces.append(normalized_face)

    return processed_faces if len(processed_faces) > 1 else processed_faces[0]

