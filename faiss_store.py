import faiss, json, os
import numpy as np

class FaissStore:
    def __init__(self, dim=768, path="data/index.faiss", meta_path="data/meta.json"):
        self.dim = dim
        self.path = path
        self.meta_path = meta_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            self.index = faiss.read_index(path)
        else:
            self.index = faiss.IndexFlatL2(dim)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metas = json.load(f)
        else:
            self.metas = []

    def add(self, vectors, metas):
        self.index.add(vectors.astype("float32"))
        self.metas.extend(metas)

    def save(self):
        faiss.write_index(self.index, self.path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metas, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.path):
            self.index = faiss.read_index(self.path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metas = json.load(f)

    def search(self, query_vector, top_k=3, document_id=None):
        if not self.metas or self.index.ntotal == 0:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1).astype("float32")
        D, I = self.index.search(query_vector, top_k)
        results = []
        for idx in I[0]:
            if idx == -1 or idx >= len(self.metas):
                continue
            meta = self.metas[idx]
            if document_id and meta.get("document_id") != document_id:
                continue
            results.append(meta)
        return results
