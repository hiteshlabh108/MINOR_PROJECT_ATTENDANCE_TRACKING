import pickle

# Path to the saved face embeddings
embeddings_path = "face_embeddings.pkl"

# Load the embeddings from the file
try:
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
        print("✅ Successfully loaded face embeddings:")
        for name, embedding in embeddings.items():
            print(f"Name: {name}, Embedding Shape: {embedding.shape}")
except FileNotFoundError:
    print(f"❌ Error: File '{embeddings_path}' not found.")
except Exception as e:
    print(f"⚠️ Error loading embeddings: {e}")
