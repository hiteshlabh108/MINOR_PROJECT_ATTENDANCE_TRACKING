import pickle
import numpy as np

# Path to the saved embeddings file
embedding_file_path = "/home/hitesh-labh/Attendance_project/face_embeddings.pkl"

# Load the embeddings from the pickle file
with open(embedding_file_path, "rb") as f:
    face_embeddings = pickle.load(f)

# Check the loaded embeddings
for name, embedding in face_embeddings.items():
    print(f"Name: {name}, Embedding: {embedding.shape}\n{embedding[:5]}...")  # Print first 5 values for preview
