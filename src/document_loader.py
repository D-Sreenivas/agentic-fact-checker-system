from langchain_core.documents import Document


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