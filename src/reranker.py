import os
from src.config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:
    """
    Evidence Reranker:
    - High-Memory / Local: CrossEncoder transformer for deep sequence-pair ranking.
    - Cloud / 512MB RAM: Reuses dense semantic similarity or TF-IDF relevance scores
      from the Retriever, incurring 0MB additional memory overhead.
    """

    def __init__(self):
        self.model = None
        is_cloud = bool(os.getenv("RENDER") or os.getenv("PORT") or os.getenv("LOW_MEMORY"))
        force_transformers = os.getenv("FORCE_TRANSFORMERS", "false").lower() in ("true", "1")

        if not is_cloud or force_transformers:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(RERANKER_MODEL)
                print(f"[RERANKER] CrossEncoder model loaded: {RERANKER_MODEL}")
            except Exception as e:
                print(f"[RERANKER NOTICE] CrossEncoder skipped ({e}). Reusing retriever scores.")
        else:
            print("[RERANKER] Cloud/low-memory mode: reusing retriever scores (0MB RAM overhead).")

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
