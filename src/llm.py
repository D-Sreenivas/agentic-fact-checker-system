from google import genai
from src.config import GEMINI_API_KEY, LLM_MODEL, FALLBACK_LLM_MODELS


client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[CLIENT INIT ERROR] {e}")


def fact_check_with_llm(claim, evidence):
    global client
    if client is None:
        if not GEMINI_API_KEY:
            return """VERDICT:
UNCERTAIN

CONFIDENCE:
0

EXPLANATION:
GEMINI_API_KEY is not configured on Render. Please go to Render Dashboard -> Environment and add GEMINI_API_KEY.

EVIDENCE_USED:
None

CITATIONS:
None"""
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            return f"ERROR: Failed to initialize Gemini client: {e}"

    evidence_text = ""

    for i, item in enumerate(evidence):

        score, document = item

        evidence_text += f"""
Evidence {i + 1}

Title:
{document.metadata.get("title", "")}

URL:
{document.metadata.get("url", "")}

Content:
{document.page_content}

Relevance Score:
{score}

-------------------------
"""

    prompt = f"""
You are an AI fact-checking agent.

Your task is to verify the following claim.

CLAIM:
{claim}

Use ONLY the provided evidence.

EVIDENCE:
{evidence_text}

Return your answer in this format:

VERDICT:
SUPPORTED / REFUTED / UNCERTAIN

CONFIDENCE:
0-100

EXPLANATION:
Explain clearly why the claim is supported, refuted,
or cannot be determined from the evidence.

EVIDENCE_USED:
List the evidence numbers used.

CITATIONS:
List the source URLs.

Do not invent evidence.
Do not invent citations.
If the evidence is insufficient, return UNCERTAIN.
"""

    models_to_try = []
    for m in [LLM_MODEL] + FALLBACK_LLM_MODELS:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"[LLM WARNING] Model {model_name} failed: {e}. Trying fallback...")
            last_error = e

    return f"ERROR: Could not verify claim with LLM. Details: {last_error}"