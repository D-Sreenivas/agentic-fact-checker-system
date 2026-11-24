from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(self, query, retrieved_documents, top_k=TOP_K_RERANK):

        if not retrieved_documents:
            return []

        documents = [
            item["document"]
            for item in retrieved_documents
        ]

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        return ranked[:top_k]