from rank_bm25 import BM25Okapi

bm25 = None


def get_chunk_text(c):
    if isinstance(c, dict):
        return c.get("chunk", c.get("text", ""))
    return str(c)


def build_bm25(chunks):
    global bm25
    if not chunks:
        return
    corpus = [get_chunk_text(chunk).split() for chunk in chunks]
    bm25 = BM25Okapi(corpus)


def bm25_search(query, chunks, top_k=5):
    global bm25
    if not chunks:
        return []
    if bm25 is None:
        build_bm25(chunks)

    tokenized_query = query.split()
    scores = list(bm25.get_scores(tokenized_query))

    ranked = sorted(
        zip(scores, chunks),
        reverse=True,
        key=lambda x: x[0]
    )

    min_s = min(scores) if scores else 0.0
    max_s = max(scores) if scores else 1.0

    results = []
    for score, doc in ranked[:top_k]:
        if max_s > min_s:
            score_pct = round(float((score - min_s) / (max_s - min_s)) * 100, 2)
        else:
            score_pct = 75.0 if max_s > 0 else 50.0

        text = get_chunk_text(doc)
        source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
        page = doc.get("page", 1) if isinstance(doc, dict) else 1

        results.append({
            "text": text,
            "score": score_pct,
            "source": source,
            "page": page
        })

    return results
