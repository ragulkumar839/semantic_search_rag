from sentence_transformers import CrossEncoder

reranker = None


def get_reranker():
    global reranker
    if reranker is None:
        try:
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            reranker = False
    return reranker


def rerank(query, candidates, top_k=5):
    """
    Pass retrieved candidates through a Cross-Encoder model to re-score
    and re-rank top candidates for optimal accuracy before passing to LLM.
    """
    if not candidates:
        return []

    model = get_reranker()
    if not model:
        return candidates[:top_k]

    try:
        pairs = [[query, c.get("text", c.get("chunk", ""))] for c in candidates]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            reverse=True,
            key=lambda x: x[0]
        )

        max_s = max(scores) if len(scores) > 0 else 1.0
        min_s = min(scores) if len(scores) > 0 else 0.0

        results = []
        for score, item in ranked[:top_k]:
            if max_s > min_s:
                norm_score = round(float((score - min_s) / (max_s - min_s)) * 100, 2)
            else:
                norm_score = 85.0

            c_copy = dict(item)
            c_copy["score"] = norm_score
            c_copy["rerank_score"] = float(score)
            results.append(c_copy)

        return results
    except Exception:
        return candidates[:top_k]
