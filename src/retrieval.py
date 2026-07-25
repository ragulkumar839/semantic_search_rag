import numpy as np
import faiss


def search(query, model, index, documents, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Convert to NumPy float32
    query_embedding = np.array(query_embedding).astype("float32")

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        doc = documents[idx]

        results.append({
            "text": doc["chunk"],
            "score": round(float(score) * 100, 2),
            "source": doc["source"],
            "page": doc["page"]
        })

    return results