import os
from src.config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:

    def __init__(self):
        self.model = None
        # On memory-constrained cloud environments (such as Render 512MB RAM),
        # loading two BERT models (SentenceTransformer + CrossEncoder) simultaneously
        # triggers Linux OOM killer (causing 502 Bad Gateway).
        # We only load CrossEncoder if NOT running in cloud or if explicit flag set.
        is_cloud = bool(os.getenv("RENDER") or os.getenv("PORT"))
        if not is_cloud:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(RERANKER_MODEL)
            except Exception as e:
                print(f"[RERANKER NOTICE] CrossEncoder skipped: {e}")

    def rerank(self, query, retrieved_documents, top_k=TOP_K_RERANK):
        if not retrieved_documents:
            return []

        # If CrossEncoder model is available, use it
        if self.model is not None:
            try:
                documents = [item["document"] for item in retrieved_documents]
                pairs = [(query, doc.page_content) for doc in documents]
                scores = self.model.predict(pairs)
                ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
                return ranked[:top_k]
            except Exception as e:
                print(f"[RERANKER ERROR] Prediction failed, using dense scores: {e}")

        # High-efficiency fallback: reuse neural semantic similarity scores from Retriever
        # (Zero extra memory overhead, runs instantly without OOM)
        sorted_docs = sorted(
            retrieved_documents,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        return [(item["score"], item["document"]) for item in sorted_docs[:top_k]]
