import os
import sys
import time
import requests
import streamlit as st

# Ensure workspace root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

st.set_page_config(
    page_title="Agentic Fact Checker | Streamlit Client",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        color: #8892b0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .status-online { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; }
    .status-offline { background-color: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Backend API Configuration
# -----------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# -----------------------------------------------------------------------------
# GET Methods: Connect Frontend (Streamlit) to Backend (FastAPI)
# -----------------------------------------------------------------------------
def fetch_backend_health(base_url: str) -> dict:
    """
    GET /api/health:
    Queries the FastAPI backend for operational status and active models.
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/health", timeout=5)
        if response.status_code == 200:
            return {"connected": True, "data": response.json()}
        return {"connected": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


def fetch_backend_examples(base_url: str) -> list:
    """
    GET /api/examples:
    Fetches curated sample verification claims from the FastAPI backend.
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/examples", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    # Fallback default examples if backend is unreachable
    return [
        {"id": "ex_1", "claim": "Australia won the 2023 ICC Men's Cricket World Cup.", "category": "Sports"},
        {"id": "ex_2", "claim": "The Eiffel Tower is located in Paris, France.", "category": "Geography"},
        {"id": "ex_3", "claim": "India won the 2023 Men's Cricket World Cup final.", "category": "Sports"},
        {"id": "ex_4", "claim": "NASA has officially confirmed extraterrestrial civilizations.", "category": "Speculation"},
    ]


def verify_claim_api(base_url: str, claim_text: str, top_k_retrieval: int = 10, top_k_rerank: int = 5) -> dict:
    """
    POST /api/verify:
    Sends the claim to the FastAPI backend verification pipeline.
    """
    payload = {
        "claim": claim_text,
        "top_k_retrieval": top_k_retrieval,
        "top_k_rerank": top_k_rerank,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/api/verify",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Backend API Error {response.status_code}: {response.text}")


# -----------------------------------------------------------------------------
# Sidebar: Backend Connection & System Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Backend Connection")

    api_url = st.text_input(
        "FastAPI Backend URL:",
        value=DEFAULT_API_URL,
        help="The URL of the FastAPI backend (e.g. http://127.0.0.1:8000 or your Render live URL).",
    )

    # Call GET /api/health
    health_info = fetch_backend_health(api_url)

    if health_info["connected"]:
        st.markdown("<span class='status-badge status-online'>🟢 Backend Connected (GET /api/health)</span>", unsafe_allow_html=True)
        data = health_info["data"]
        st.caption(f"**Status:** `{data.get('status', 'operational')}`")
        st.caption(f"**LLM Model:** `{data.get('llm_model', 'N/A')}`")
        st.caption(f"**Retriever:** `{data.get('embedding_model', 'N/A')}`")
        st.caption(f"**Reranker:** `{data.get('reranker_model', 'N/A')}`")
    else:
        st.markdown("<span class='status-badge status-offline'>🔴 Backend Offline</span>", unsafe_allow_html=True)
        st.caption(f"Could not connect to `{api_url}`: {health_info.get('error', 'Unknown')}")
        st.info("💡 Start the backend with `python app/main.py` in your terminal.")

    st.divider()

    st.subheader("💡 Sample Claims (GET /api/examples)")
    examples = fetch_backend_examples(api_url)

    for ex in examples:
        claim_text = ex.get("claim", "")
        category = ex.get("category", "General")
        if st.button(f"📌 [{category}] {claim_text[:40]}...", key=f"ex_{ex.get('id', claim_text)}", use_container_width=True):
            st.session_state["claim_input"] = claim_text


# -----------------------------------------------------------------------------
# Main Application Interface
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>🔎 Agentic Fact-Checking System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Streamlit Frontend connected to FastAPI Backend over REST API (GET / POST).</div>", unsafe_allow_html=True)

# Input Box
default_claim = st.session_state.get("claim_input", "")
claim = st.text_area(
    "Enter a statement or claim to verify:",
    value=default_claim,
    height=100,
    placeholder="Example: The Eiffel Tower is located in Paris, France.",
)

col1, col2, col3 = st.columns([1.5, 1, 4])
with col1:
    verify_clicked = st.button("🔍 Verify Claim with AI", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state["claim_input"] = ""
        st.rerun()

# -----------------------------------------------------------------------------
# Execute Verification (POST /api/verify)
# -----------------------------------------------------------------------------
if verify_clicked:
    claim_clean = claim.strip()
    if not claim_clean:
        st.warning("⚠️ Please enter a claim to verify.")
    else:
        try:
            start_time = time.time()
            with st.spinner("Calling backend pipeline (Search ➔ Retrieval ➔ Rerank ➔ Gemini LLM)..."):
                # Check if backend is reachable
                if health_info["connected"]:
                    # Method 1: Connected via HTTP POST to FastAPI backend
                    result = verify_claim_api(api_url, claim_clean)
                else:
                    # Method 2: Fallback to direct agent execution if backend is offline
                    st.info("ℹ️ Backend server offline: Running directly via local FactCheckingAgent...")
                    from src.agent import FactCheckingAgent
                    agent = FactCheckingAgent()
                    result = agent.run(claim_clean)

            elapsed = round(time.time() - start_time, 2)

            # Extract fields
            verdict = result.get("verdict", "UNCERTAIN").upper()
            confidence = result.get("confidence", 0)
            explanation = result.get("explanation", "")
            citations = result.get("citations", [])
            evidence_items = result.get("evidence", []) or result.get("structured_evidence", [])

            st.divider()
            st.subheader("📋 Verification Verdict")

            # Colored verdict banner
            if verdict == "SUPPORTED":
                st.success(f"### ✅ Verdict: SUPPORTED")
            elif verdict == "REFUTED":
                st.error(f"### ❌ Verdict: REFUTED")
            elif verdict == "UNCERTAIN":
                st.warning(f"### ⚠️ Verdict: UNCERTAIN")
            else:
                st.info(f"### ℹ️ Verdict: {verdict}")

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Confidence", f"{confidence}%")
            with m2:
                st.metric("Processing Time", f"{elapsed}s")
            with m3:
                st.metric("Evidence Sources", len(evidence_items))

            st.progress(min(max(confidence / 100.0, 0.0), 1.0))

            # Explanation Box
            with st.container():
                st.markdown("#### 📝 Detailed Synthesis & Explanation")
                st.write(explanation)

            # Citations List
            if citations:
                with st.expander("🔗 Source Citations", expanded=True):
                    for cite in citations:
                        st.markdown(f"- [{cite}]({cite})")

            # Retrieved Evidence Cards
            if evidence_items:
                st.divider()
                st.subheader(f"📚 Retrieved & Reranked Evidence ({len(evidence_items)} sources)")

                for idx, item in enumerate(evidence_items, 1):
                    # Handle both dictionary and tuple formats
                    if isinstance(item, dict):
                        title = item.get("title", f"Evidence #{idx}")
                        url = item.get("url", "#")
                        score = item.get("score", 0.0)
                        content = item.get("content", "")
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        score, doc = item
                        title = getattr(doc, "metadata", {}).get("title", f"Evidence #{idx}")
                        url = getattr(doc, "metadata", {}).get("url", "#")
                        content = getattr(doc, "page_content", "")
                    else:
                        continue

                    with st.expander(f"#{idx} — {title} (Score: {score:.4f})", expanded=(idx <= 2)):
                        st.markdown(f"**URL:** [{url}]({url})")
                        st.markdown(f"**Relevance Score:** `{score:.4f}`")
                        st.markdown("**Evidence Snippet:**")
                        st.info(content)

        except Exception as e:
            st.error(f"An error occurred while verifying the claim: {e}")
