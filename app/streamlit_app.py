import os
import re
import sys
import time
import requests
import streamlit as st

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import FactCheckingAgent
from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    RERANKER_MODEL,
    TOP_K_RERANK,
    TOP_K_RETRIEVAL,
)

st.set_page_config(
    page_title="Agentic Fact Checker",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished aesthetics
st.markdown(
    """
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
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    .get-box {
        background-color: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .http-method {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        background-color: #2563eb;
        color: white;
        margin-right: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing retrieval & reranking models...")
def get_agent():
    """Cache the fact checking agent so embedding & reranking models load only once."""
    return FactCheckingAgent()


# FastAPI Backend Base URL
DEFAULT_API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

# Sidebar with System Information & GET Methods
with st.sidebar:
    st.header("⚙️ Architecture")
    st.markdown("""
    **Multi-Stage Fact Verification:**
    1. 🌐 **Web Retrieval** (`DDGS` + Wikipedia)
    2. 🧠 **Dense Semantic / TF-IDF Search**
    3. ⚖️ **Reranking Engine**
    4. 🤖 **Gemini Verification** (`gemini-3.5-flash`)
    """)
    st.divider()

    st.subheader("Model Configuration")
    st.caption(f"**Primary LLM:** `{LLM_MODEL}`")
    st.caption(f"**Embedder:** `{EMBEDDING_MODEL}`")
    st.caption(f"**Reranker:** `{RERANKER_MODEL}`")
    st.caption(f"**Retrieval Candidates:** `{TOP_K_RETRIEVAL}`")
    st.caption(f"**Reranked Evidence:** `{TOP_K_RERANK}`")

    st.divider()
    st.subheader("💡 Try Example Claims")

    examples = [
        "The Eiffel Tower is located in Paris, France.",
        "Australia won the 2023 ICC Men's Cricket World Cup.",
        "India won the 2023 Men's Cricket World Cup final.",
        "Humans landed on Mars in 2020.",
        "NASA has officially confirmed the discovery of extraterrestrial civilizations.",
    ]

    for ex in examples:
        if st.button(ex, key=f"btn_{ex}"):
            st.session_state["claim_input"] = ex


# -------------------------------------------------------------
# Handle GET Method Query Parameters (e.g. ?claim=...)
# -------------------------------------------------------------
url_claim = st.query_params.get("claim", "")
if url_claim and "claim_input" not in st.session_state:
    st.session_state["claim_input"] = url_claim


# Main Interface
st.markdown("<div class='main-title'>🔎 Agentic Fact-Checking System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Verify claims automatically using real-time search, dense retrieval, neural reranking, and Gemini.</div>",
    unsafe_allow_html=True,
)

# Tabs: Fact Checker and GET Method Explorer
tab_verify, tab_get_api = st.tabs(["🔍 Fact Checker", "📡 FastAPI GET Methods Explorer"])

with tab_verify:
    # Input Box
    default_claim = st.session_state.get("claim_input", "")
    claim = st.text_area(
        "Enter a statement or claim to verify:",
        value=default_claim,
        height=100,
        placeholder="Example: The Eiffel Tower is in Paris, France.",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        verify_clicked = st.button("🔍 Fact Check", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=False):
            st.session_state["claim_input"] = ""
            st.query_params.clear()
            st.rerun()

    def parse_llm_response(text):
        """Parse verdict, confidence, explanation, and citations from response."""
        verdict_match = re.search(r"VERDICT:\s*([A-Za-z]+)", text, re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
        explanation_match = re.search(
            r"EXPLANATION:\s*(.*?)(?=(?:EVIDENCE_USED:|CITATIONS:|$))",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        citations_match = re.search(r"CITATIONS:\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)

        verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
        confidence = int(confidence_match.group(1)) if confidence_match else None
        explanation = explanation_match.group(1).strip() if explanation_match else text
        citations = citations_match.group(1).strip() if citations_match else ""

        return verdict, confidence, explanation, citations

    if verify_clicked:
        if not claim.strip():
            st.warning("⚠️ Please enter a claim to verify.")
        else:
            try:
                start_time = time.time()
                with st.spinner("Agent is searching, retrieving, reranking, and verifying evidence..."):
                    agent = get_agent()
                    result = agent.run(claim.strip())

                elapsed = round(time.time() - start_time, 2)
                verdict_text = result.get("result", "")
                evidence_list = result.get("evidence", [])

                verdict, confidence, explanation, citations = parse_llm_response(verdict_text)

                st.divider()
                st.subheader("📋 Verification Verdict")

                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    if verdict == "SUPPORTED":
                        st.success("### ✅ SUPPORTED")
                    elif verdict == "REFUTED":
                        st.error("### ❌ REFUTED")
                    elif verdict == "UNCERTAIN":
                        st.warning("### ⚠️ UNCERTAIN")
                    else:
                        st.info(f"### ℹ️ {verdict}")
                with m2:
                    st.metric("Confidence", f"{confidence or 0}%")
                with m3:
                    st.metric("Processing Time", f"{elapsed}s")

                # Explanation Box
                with st.container():
                    st.markdown("#### 📝 Detailed Explanation")
                    st.write(explanation)

                if citations:
                    with st.expander("🔗 Citations & Sources", expanded=True):
                        st.write(citations)

                # Evidence Section
                st.divider()
                st.subheader(f"📚 Retrieved & Reranked Evidence ({len(evidence_list)} sources)")

                if not evidence_list:
                    st.info("No evidence documents were retrieved for this claim.")
                else:
                    for idx, (score, doc) in enumerate(evidence_list, 1):
                        title = doc.metadata.get("title", f"Evidence #{idx}")
                        url = doc.metadata.get("url", "#")
                        with st.expander(f"#{idx} — {title} (Relevance Score: {score:.4f})", expanded=(idx <= 2)):
                            st.markdown(f"**Source URL:** [{url}]({url})")
                            st.markdown(f"**Relevance Score:** `{score:.4f}`")
                            st.markdown("**Content Snippet:**")
                            st.info(doc.page_content)

            except Exception as e:
                st.error(f"An error occurred while verifying the claim: {e}")

# -------------------------------------------------------------
# Tab 2: GET Methods Explorer
# -------------------------------------------------------------
with tab_get_api:
    st.markdown("### 🌐 FastAPI GET Methods Explorer")
    st.markdown(
        "This section demonstrates how to consume the backend using **HTTP GET methods** "
        "(via URL parameters, browser address bar, or Python `requests.get`)."
    )

    api_url = st.text_input("FastAPI Server Base URL:", value=DEFAULT_API_URL)

    # 1. GET /api/health
    st.markdown("---")
    st.markdown("#### 1. `<span class='http-method'>GET</span> /api/health`", unsafe_allow_html=True)
    st.caption("Fetches operational status, active embedding model, reranker model, and LLM.")

    if st.button("Execute GET /api/health", key="btn_get_health"):
        try:
            with st.spinner("Calling GET /api/health..."):
                t0 = time.time()
                resp = requests.get(f"{api_url.rstrip('/')}/api/health", timeout=10)
                dur = round(time.time() - t0, 3)

            st.write(f"**Status Code:** `{resp.status_code} OK` | **Latency:** `{dur}s`")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Failed to connect to FastAPI at {api_url}: {e}")

    # 2. GET /api/examples
    st.markdown("---")
    st.markdown("#### 2. `<span class='http-method'>GET</span> /api/examples`", unsafe_allow_html=True)
    st.caption("Returns the curated collection of fact-checking test claims.")

    if st.button("Execute GET /api/examples", key="btn_get_examples"):
        try:
            with st.spinner("Calling GET /api/examples..."):
                t0 = time.time()
                resp = requests.get(f"{api_url.rstrip('/')}/api/examples", timeout=10)
                dur = round(time.time() - t0, 3)

            st.write(f"**Status Code:** `{resp.status_code} OK` | **Latency:** `{dur}s`")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Failed to connect to FastAPI at {api_url}: {e}")

    # 3. GET /api/verify?claim=...
    st.markdown("---")
    st.markdown("#### 3. `<span class='http-method'>GET</span> /api/verify?claim=...`", unsafe_allow_html=True)
    st.caption("Verifies a claim directly through a GET request with URL query parameter.")

    test_query = st.text_input(
        "Claim Query Parameter (`?claim=`):",
        value="The Earth revolves around the Sun in 365 days.",
        key="get_claim_param",
    )

    encoded_url = f"{api_url.rstrip('/')}/api/verify?claim={requests.utils.quote(test_query)}"
    st.code(f"GET {encoded_url}", language="http")

    if st.button("Execute GET /api/verify", key="btn_get_verify"):
        try:
            with st.spinner("Sending GET request to FastAPI pipeline..."):
                t0 = time.time()
                resp = requests.get(encoded_url, timeout=60)
                dur = round(time.time() - t0, 2)

            st.write(f"**Status Code:** `{resp.status_code} OK` | **Processing Time:** `{dur}s`")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Error calling GET /api/verify: {e}")

    # 4. Streamlit URL GET parameter support
    st.markdown("---")
    st.markdown("#### 4. Streamlit URL Query Parameters (`st.query_params`)")
    st.markdown(
        """
        You can also pass claims directly to this Streamlit interface via URL GET parameters:
        ```text
        http://localhost:8501/?claim=The+Eiffel+Tower+is+in+Paris
        ```
        When accessed with a `?claim=` parameter in the browser, Streamlit automatically populates the input field via `st.query_params`!
        """
    )
