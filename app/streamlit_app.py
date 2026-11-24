import streamlit as st
import sys
import os
import re

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import FactCheckingAgent
from src.config import EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK

st.set_page_config(
    page_title="Agentic Fact Checker",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished aesthetics
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
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    .verdict-box {
        padding: 1.25rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .verdict-supported {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
    }
    
    .verdict-refuted {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
    }
    
    .verdict-uncertain {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
    }
    
    .evidence-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-green { background-color: #10b981; color: white; }
    .badge-red { background-color: #ef4444; color: white; }
    .badge-yellow { background-color: #f59e0b; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing retrieval & reranking models...")
def get_agent():
    """Cache the fact checking agent so embedding & reranking models load only once."""
    return FactCheckingAgent()


# Sidebar with System Information
with st.sidebar:
    st.header("⚙️ Architecture")
    st.markdown("""
    **Multi-Stage Fact Verification:**
    1. 🌐 **Web Retrieval** (`DDGS`)
    2. 🧠 **Dense Semantic Search** (`SentenceTransformer`)
    3. ⚖️ **Cross-Encoder Reranking** (`MiniLM-L6`)
    4. 🤖 **Gemini Verification** (Evidence Reasoning)
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
        "India won the 2023 Men's Cricket World Cup.",
        "Humans landed on Mars in 2020."
    ]
    
    for ex in examples:
        if st.button(ex, key=f"btn_{ex}"):
            st.session_state["claim_input"] = ex


# Main Interface
st.markdown("<div class='main-title'>🔎 Agentic Fact-Checking System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Verify claims automatically using real-time search, dense retrieval, neural reranking, and Gemini.</div>", unsafe_allow_html=True)

# Input Box
default_claim = st.session_state.get("claim_input", "")
claim = st.text_area(
    "Enter a statement or claim to verify:",
    value=default_claim,
    height=100,
    placeholder="Example: India won the 2023 Cricket World Cup."
)

col1, col2 = st.columns([1, 5])
with col1:
    verify_clicked = st.button("🔍 Fact Check", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=False):
        st.session_state["claim_input"] = ""
        st.rerun()

def parse_llm_response(text):
    """Parse verdict, confidence, explanation, and citations from response."""
    verdict_match = re.search(r"VERDICT:\s*([A-Za-z]+)", text, re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
    explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?=(?:EVIDENCE_USED:|CITATIONS:|$))", text, re.IGNORECASE | re.DOTALL)
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
            with st.spinner("Agent is searching, retrieving, reranking, and verifying evidence..."):
                agent = get_agent()
                result = agent.run(claim.strip())

            verdict_text = result.get("result", "")
            evidence_list = result.get("evidence", [])
            
            verdict, confidence, explanation, citations = parse_llm_response(verdict_text)
            
            st.divider()
            st.subheader("📋 Verification Verdict")
            
            # Display colored verdict banner
            if verdict == "SUPPORTED":
                st.success(f"### ✅ Verdict: SUPPORTED")
            elif verdict == "REFUTED":
                st.error(f"### ❌ Verdict: REFUTED")
            elif verdict == "UNCERTAIN":
                st.warning(f"### ⚠️ Verdict: UNCERTAIN")
            else:
                st.info(f"### ℹ️ Verdict: {verdict}")

            # Confidence Metric
            if confidence is not None:
                st.markdown(f"**Confidence Level:** `{confidence}%`")
                st.progress(min(max(confidence / 100.0, 0.0), 1.0))

            # Explanation Box
            with st.container():
                st.markdown("#### 📝 Explanation")
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