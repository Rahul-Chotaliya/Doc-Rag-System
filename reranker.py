from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"[Warning] Could not load CrossEncoder ({e}), fallback to simple scoring")
            self.model = None

    def rerank(self, query, docs):
        if not docs:
            return []
        if self.model is None:
            return docs
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in reranked]
