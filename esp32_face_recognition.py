import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import dlib

# Load Dlib models
shape_predictor_path = "/home/hitesh-labh/Attendance_project/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "/home/hitesh-labh/Attendance_project/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Load stored face embeddings from file
embeddings_path = "/home/hitesh-labh/Attendance_project/face_embeddings.pkl"
with open(embeddings_path, "rb") as f:
    face_database = pickle.load(f)

def get_embedding(image):
    """Extract 128D face embedding from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    shape = predictor(image, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    
    return np.array(face_descriptor)

def recognize_face(input_image):
    """Match face using cosine similarity"""
    embedding = get_embedding(input_image)

    if embedding is None:
        print("❌ No face detected.")
        return None

    best_match = None
    highest_similarity = -1  # Start with lowest possible cosine similarity

    for name, stored_embedding in face_database.items():
        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]  # Compute similarity
        print(f"Similarity with {name}: {similarity}")

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name

    if highest_similarity > 0.6:  # Set a threshold (tweak based on performance)
        print(f"✅ Matched with: {best_match} (Similarity: {highest_similarity:.2f})")
        return best_match
    else:
        print("❌ No matching face found.")
        return None

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Draw rectangle around the face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize the face
        recognized_name = recognize_face(frame)
        if recognized_name:
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import dlib

# Load Dlib models
shape_predictor_path = "/home/hitesh-labh/Attendance_project/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "/home/hitesh-labh/Attendance_project/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Load stored face embeddings from file
embeddings_path = "/home/hitesh-labh/Attendance_project/face_embeddings.pkl"
with open(embeddings_path, "rb") as f:
    face_database = pickle.load(f)

def get_embedding(image):
    """Extract 128D face embedding from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    shape = predictor(image, faces[0])
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    
    return np.array(face_descriptor)

def recognize_face(input_image):
    """Match face using cosine similarity"""
    embedding = get_embedding(input_image)

    if embedding is None:
        print("❌ No face detected.")
        return None

    best_match = None
    highest_similarity = -1  # Start with lowest possible cosine similarity

    for name, stored_embedding in face_database.items():
        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]  # Compute similarity
        print(f"Similarity with {name}: {similarity}")

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name

    if highest_similarity > 0.6:  # Set a threshold (tweak based on performance)
        print(f"✅ Matched with: {best_match} (Similarity: {highest_similarity:.2f})")
        return best_match
    else:
        print("❌ No matching face found.")
        return None

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Draw rectangle around the face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize the face
        recognized_name = recognize_face(frame)

    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
