from src.retrieval import search
from src.bm25_search import bm25_search
from src.reranker import rerank


def hybrid_search(
    query,
    model,
    index,
    chunks,
    top_k=5,
    source_filter=None
):
    """
    Production Hybrid Search Engine:
    1. FAISS Dense Vector Retrieval (all-MiniLM-L6-v2)
    2. BM25 Sparse Keyword Retrieval (BM25Okapi)
    3. Metadata Filtering (Source document isolation)
    4. Cross-Encoder Re-ranker (ms-marco-MiniLM-L-6-v2)
    """
    if not chunks:
        return []

    candidate_k = max(top_k * 2, 10)

    semantic = search(
        query,
        model,
        index,
        chunks,
        candidate_k
    )

    keyword = bm25_search(
        query,
        chunks,
        candidate_k
    )

    merged = []
    seen = set()

    for item in semantic + keyword:
        key = (
            item.get("source", ""),
            item.get("page", 1),
            item.get("text", "")
        )

        if key not in seen:
            seen.add(key)

            # Metadata filtering by source filename if filter provided
            if source_filter and source_filter != "All Documents":
                doc_source = str(item.get("source", "")).lower()
                filter_term = str(source_filter).lower()
                if filter_term not in doc_source:
                    continue

            merged.append(item)

    # Re-rank candidates using Cross-Encoder
    final_results = rerank(query, merged, top_k=top_k)
    return final_results
