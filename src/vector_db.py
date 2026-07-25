import os
import pickle
import faiss
import numpy as np

def create_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index

def save_index(index, chunks):
    os.makedirs("models", exist_ok=True)

    faiss.write_index(index, "models/faiss.index")

    with open("models/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index():

    if os.path.exists("models/faiss.index") and os.path.exists("models/chunks.pkl"):

        index = faiss.read_index("models/faiss.index")

        with open("models/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        return index, chunks

    return None, None