import os
import numpy as np
from src.config import EMBEDDING_MODEL, TOP_K_RETRIEVAL


class Retriever:
    """
    Hybrid Evidence Retriever:
    - On local machines with sufficient RAM: Uses dense neural embeddings via SentenceTransformers.
    - On cloud/containerized environments with 512MB RAM (e.g. Render Free Tier):
      Uses high-speed, zero-OOM TF-IDF n-gram vectorization (<5MB RAM footprint)
      to eliminate memory crashes (502 Bad Gateway) while maintaining strong retrieval quality.
    """

    def __init__(self):
        self.model = None
        self.use_tfidf = False

        is_cloud = bool(os.getenv("RENDER") or os.getenv("PORT") or os.getenv("LOW_MEMORY"))
        force_transformers = os.getenv("FORCE_TRANSFORMERS", "false").lower() in ("true", "1")

        if not is_cloud or force_transformers:
            try:
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                print(f"[RETRIEVER] Dense neural model loaded: {EMBEDDING_MODEL}")
            except Exception as e:
                print(f"[RETRIEVER NOTICE] Dense neural model unavailable ({e}). Using fast TF-IDF.")
                self.use_tfidf = True
        else:
            print("[RETRIEVER] Cloud / low-memory environment detected. Using fast TF-IDF retriever.")
            self.use_tfidf = True

    def retrieve(self, query: str, documents: list, top_k: int = TOP_K_RETRIEVAL):
        if not documents:
            return []

        # Strategy 1: Dense Neural Encoding (Local / High-Memory)
        if self.model is not None and not self.use_tfidf:
            try:
                import torch
                texts = [doc.page_content for doc in documents]
                with torch.inference_mode():
                    document_embeddings = self.model.encode(texts, normalize_embeddings=True)
                    query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

                scores = np.dot(document_embeddings, query_embedding)
                ranked_indices = np.argsort(scores)[::-1]

                results = []
                for index in ranked_indices[:top_k]:
                    results.append({
                        "document": documents[index],
                        "score": float(scores[index]),
                    })
                return results
            except Exception as e:
                print(f"[RETRIEVER ERROR] Neural encoding error: {e}. Falling back to TF-IDF.")

        # Strategy 2: Ultra-lightweight TF-IDF n-gram retrieval (<5MB RAM, instant response)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [doc.page_content for doc in documents]
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([query] + texts)
            query_vec = tfidf_matrix[0]
            doc_vecs = tfidf_matrix[1:]
            scores = (doc_vecs * query_vec.T).toarray().flatten()
        except Exception as e:
            print(f"[RETRIEVER TFIDF NOTE] TF-IDF scoring exception: {e}")
            scores = np.ones(len(documents), dtype=float)

        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for index in ranked_indices[:top_k]:
            results.append({
                "document": documents[index],
                "score": float(scores[index]),
            })

        return results
