import re
from src.search_tool import search_web
from src.document_loader import convert_search_results_to_documents
from src.retriever import Retriever
from src.reranker import Reranker
from src.llm import fact_check_with_llm
from src.config import TOP_K_RETRIEVAL, TOP_K_RERANK


def parse_llm_response(text: str) -> dict:
    """Parse raw LLM response into structured fact-checking attributes."""
    if text.startswith("ERROR:"):
        return {
            "verdict": "ERROR",
            "confidence": 0,
            "explanation": text,
            "evidence_used": "",
            "citations": [],
        }

    verdict_match = re.search(r"VERDICT:\s*([A-Za-z]+)", text, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
    explanation_match = re.search(
        r"EXPLANATION:\s*(.*?)(?=(?:EVIDENCE_USED:|CITATIONS:|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    evidence_used_match = re.search(
        r"EVIDENCE_USED:\s*(.*?)(?=(?:CITATIONS:|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    citations_match = re.search(r"CITATIONS:\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)

    verdict = verdict_match.group(1).upper() if verdict_match else "UNCERTAIN"
    confidence = int(confidence_match.group(1)) if confidence_match else 0
    explanation = explanation_match.group(1).strip() if explanation_match else text.strip()
    evidence_used_raw = evidence_used_match.group(1).strip() if evidence_used_match else ""
    citations_raw = citations_match.group(1).strip() if citations_match else ""

    # Extract all HTTP/HTTPS links
    extracted_urls = re.findall(r"https?://[^\s)\]]+", citations_raw)
    if extracted_urls:
        citations = list(dict.fromkeys(extracted_urls))  # deduplicate preserving order
    elif citations_raw and citations_raw.lower() != "none":
        citations = [
            line.strip("- *").strip()
            for line in citations_raw.splitlines()
            if line.strip() and line.strip().lower() != "none"
        ]
    else:
        citations = []

    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "evidence_used": evidence_used_raw,
        "citations": citations,
    }


class FactCheckingAgent:

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()

    def run(self, claim: str, top_k_retrieval: int = None, top_k_rerank: int = None) -> dict:
        k_retrieve = top_k_retrieval or TOP_K_RETRIEVAL
        k_rerank = top_k_rerank or TOP_K_RERANK

        print("\n[AGENT] Analyzing claim...")

        # --------------------------------
        # STEP 1: WEB SEARCH
        # --------------------------------
        print("[AGENT] Searching for evidence...")
        search_results = search_web(claim, max_results=k_retrieve)

        if not search_results:
            fallback_text = """VERDICT:
UNCERTAIN

CONFIDENCE:
0

EXPLANATION:
No reliable web evidence could be retrieved for this claim.

EVIDENCE_USED:
None

CITATIONS:
None"""
            parsed = parse_llm_response(fallback_text)
            return {
                "claim": claim,
                "verdict": parsed["verdict"],
                "confidence": parsed["confidence"],
                "explanation": parsed["explanation"],
                "evidence_used": parsed["evidence_used"],
                "citations": parsed["citations"],
                "result": fallback_text,
                "evidence": [],
                "structured_evidence": [],
            }

        # --------------------------------
        # STEP 2: CONVERT TO DOCUMENTS
        # --------------------------------
        print(f"[AGENT] Found {len(search_results)} search results.")
        documents = convert_search_results_to_documents(search_results)

        # --------------------------------
        # STEP 3: RETRIEVAL
        # --------------------------------
        print("[AGENT] Retrieving relevant evidence...")
        retrieved = self.retriever.retrieve(claim, documents, top_k=k_retrieve)

        # --------------------------------
        # STEP 4: RERANKING
        # --------------------------------
        print("[AGENT] Reranking evidence...")
        reranked = self.reranker.rerank(claim, retrieved, top_k=k_rerank)

        # --------------------------------
        # STEP 5: LLM FACT CHECK
        # --------------------------------
        print("[AGENT] Verifying claim with LLM...")
        llm_response_text = fact_check_with_llm(claim, reranked)

        parsed = parse_llm_response(llm_response_text)

        structured_evidence = []
        for score, doc in reranked:
            structured_evidence.append({
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "score": round(float(score), 4),
                "content": doc.page_content,
            })

        # --------------------------------
        # FINAL RESULT
        # --------------------------------
        return {
            "claim": claim,
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "explanation": parsed["explanation"],
            "evidence_used": parsed["evidence_used"],
            "citations": parsed["citations"],
            "result": llm_response_text,
            "evidence": reranked,
            "structured_evidence": structured_evidence,
        }