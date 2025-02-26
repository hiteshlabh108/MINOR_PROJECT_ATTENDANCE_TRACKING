import dlib
import cv2
import numpy as np
import pickle
import os
import re  # Import regex for cleaning names
import logging
import sys

# Load Dlib's pretrained models
shape_predictor_path = "/home/hitesh-labh/Attendance_project/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "/home/hitesh-labh/Attendance_project/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor(shape_predictor_path)  # Landmark predictor
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)  # ResNet face embeddings

def get_embedding(image):
    """
    Detects a face in the image, aligns it, and extracts its 128D embedding.
    Returns a NumPy array representing the face.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces

    if len(faces) == 0:
        return None  # No face detected

    shape = predictor(image, faces[0])  # Detect facial landmarks
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)  # Extract 128D embedding
    
    return np.array(face_descriptor)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Path to folder containing training images
dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/home/hitesh-labh/Attendance_project/dataset/"
database = {}

# Temporary storage for embeddings
embedding_storage = {}

# Loop through images in dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = re.sub(r'\d+$', '', os.path.splitext(filename)[0]).strip().lower()  # Extract only the person's name, removing trailing numbers and convert to lowercase

        image_path = os.path.join(dataset_path, filename)
        try:
            image = cv2.imread(image_path)  # Load image
            if image is None:
                logging.error(f"❌ Unable to read image {image_path}")
                continue
            embedding = get_embedding(image)  # Extract face embedding

            if embedding is not None:
                if name not in embedding_storage:
                    embedding_storage[name] = []  # Initialize list for new name
                embedding_storage[name].append(embedding)  # Append the embedding to the list
            else:
                logging.warning(f"❌ No face detected in {filename}")

        except Exception as e:
            logging.error(f"⚠️ Error processing {filename}: {e}")  # Ensure error handling

# Compute the average embedding for each person
logging.info("\n✅ Averaged Face Embeddings:")
for name, embeddings in embedding_storage.items():
    avg_embedding = np.mean(embeddings, axis=0)  # Compute average embedding
    database[name] = avg_embedding
    logging.info(f"✅ Final embedding saved for {name} (from {len(embeddings)} images)")

# Save the averaged embeddings to a file
output_path = "/home/hitesh-labh/Attendance_project/face_embeddings.pkl"
try:
    with open(output_path, "wb") as f:
        pickle.dump(database, f)
    logging.info("\n✅ All Averaged Face Embeddings Saved Successfully!")
except Exception as e:
    logging.error(f"⚠️ Failed to save embeddings: {e}")
