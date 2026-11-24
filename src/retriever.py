from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import EMBEDDING_MODEL, TOP_K_RETRIEVAL


class Retriever:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def retrieve(self, query, documents, top_k=TOP_K_RETRIEVAL):

        if not documents:
            return []

        texts = [
            doc.page_content
            for doc in documents
        ]

        document_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = np.dot(
            document_embeddings,
            query_embedding
        )

        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "document": documents[index],
                "score": float(scores[index])
            })

        return results