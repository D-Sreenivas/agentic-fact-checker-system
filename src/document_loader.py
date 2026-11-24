try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        """Lightweight fallback Document container when langchain is not installed."""
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


def convert_search_results_to_documents(results):

    documents = []

    for result in results:

        text = (
            f"Title: {result['title']}\n"
            f"Source: {result['url']}\n"
            f"Content: {result['snippet']}"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "title": result["title"],
                    "url": result["url"]
                }
            )
        )

    return documents