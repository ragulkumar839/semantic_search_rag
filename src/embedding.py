from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):

    texts = [chunk["chunk"] for chunk in chunks]

    embeddings = model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    return embeddings