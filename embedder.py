from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs
