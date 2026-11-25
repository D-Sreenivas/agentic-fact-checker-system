import asyncio
import os
import socket
import sys
import time

# ---------------------------------------------------------------------------
# Cloud Deployment Adapter (Render, Heroku, etc.)
# If running in cloud environment where PORT is provided, automatically
# re-route any 127.0.0.1 bindings to 0.0.0.0:$PORT so Render health checks succeed
# even if Start Command was passed with '--host 127.0.0.1 --port 8000'.
# ---------------------------------------------------------------------------
_orig_socket_bind = socket.socket.bind
def _cloud_safe_bind(self, address):
    if (os.environ.get("RENDER") or os.environ.get("PORT")) and isinstance(address, tuple) and len(address) == 2:
        host, port = address
        cloud_port = int(os.environ.get("PORT", port))
        if host in ("127.0.0.1", "localhost", ""):
            print(f"[CLOUD ADAPTER] Auto-rebinding from {host}:{port} -> 0.0.0.0:{cloud_port}")
            address = ("0.0.0.0", cloud_port)
    return _orig_socket_bind(self, address)
socket.socket.bind = _cloud_safe_bind

_orig_create_server = asyncio.AbstractEventLoop.create_server
def _cloud_safe_create_server(self, protocol_factory, host=None, port=None, *args, **kwargs):
    if os.environ.get("RENDER") or os.environ.get("PORT"):
        cloud_port = int(os.environ.get("PORT", port if port else 8000))
        if host in ("127.0.0.1", "localhost", None, ""):
            print(f"[CLOUD ADAPTER] create_server: auto-rebinding from {host}:{port} -> 0.0.0.0:{cloud_port}")
            host = "0.0.0.0"
            port = cloud_port
    return _orig_create_server(self, protocol_factory, host=host, port=port, *args, **kwargs)
asyncio.AbstractEventLoop.create_server = _cloud_safe_create_server

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.agent import FactCheckingAgent
from src.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    RERANKER_MODEL,
    TOP_K_RERANK,
    TOP_K_RETRIEVAL,
)

# Global agent instance
agent_instance: Optional[FactCheckingAgent] = None


async def _pipe_stream(reader, writer):
    """Pipes raw bytes between two streams."""
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _proxy_client_handler(client_reader, client_writer, target_port=8000):
    """Bridges incoming cloud requests to internal 127.0.0.1 listener."""
    try:
        remote_reader, remote_writer = await asyncio.open_connection("127.0.0.1", target_port)
    except Exception:
        try:
            client_writer.close()
        except Exception:
            pass
        return
    await asyncio.gather(
        _pipe_stream(client_reader, remote_writer),
        _pipe_stream(remote_reader, client_writer),
        return_exceptions=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler."""
    global agent_instance

    # -------------------------------------------------------------
    # Render / Cloud Network Bridge:
    # If Uvicorn was invoked with '--host 127.0.0.1 --port 8000',
    # Render's external port scanner on 0.0.0.0:$PORT will fail.
    # We automatically bridge 0.0.0.0:$PORT -> 127.0.0.1:8000.
    # -------------------------------------------------------------
    proxy_servers = []
    target_ports = set()
    if os.getenv("PORT"):
        try:
            target_ports.add(int(os.environ["PORT"]))
        except ValueError:
            pass
    if os.getenv("RENDER"):
        target_ports.add(10000)

    for p in target_ports:
        if p != 8000:
            try:
                srv = await asyncio.start_server(
                    lambda r, w, port=8000: _proxy_client_handler(r, w, port),
                    host="0.0.0.0",
                    port=p,
                )
                proxy_servers.append(srv)
                print(f"[RENDER BRIDGE] Successfully opened 0.0.0.0:{p} -> forwarding to 127.0.0.1:8000")
            except Exception as e:
                print(f"[RENDER BRIDGE NOTE] Port {p} bridge: {e}")

    # On resource-constrained environments (like Render Free 512MB),
    # lazy loading ensures the server boots and binds port instantly without OOM.
    preload = os.getenv("PRELOAD_MODELS", "false").lower() in ("true", "1")
    if preload:
        print("[FASTAPI STARTUP] Preloading FactCheckingAgent & neural models...")
        try:
            agent_instance = await asyncio.to_thread(FactCheckingAgent)
            print("[FASTAPI STARTUP] FactCheckingAgent loaded successfully!")
        except Exception as e:
            print(f"[FASTAPI STARTUP WARNING] Failed to preload agent: {e}")
    else:
        print("[FASTAPI STARTUP] Lazy loading enabled. Models will load on first request.")

    yield

    print("[FASTAPI SHUTDOWN] Shutting down agent service...")
    for srv in proxy_servers:
        try:
            srv.close()
            await srv.wait_closed()
        except Exception:
            pass


app = FastAPI(
    title="Agentic Fact-Checker API",
    description="Production-grade AI pipeline for multi-stage claim verification using Web Retrieval, Neural Reranking, and Gemini.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external frontends or integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------

class FactCheckRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="The statement or factual claim to verify.",
        example="The Eiffel Tower is located in Paris, France.",
    )
    top_k_retrieval: Optional[int] = Field(
        default=TOP_K_RETRIEVAL,
        ge=1,
        le=20,
        description="Number of candidate web search documents to retrieve.",
    )
    top_k_rerank: Optional[int] = Field(
        default=TOP_K_RERANK,
        ge=1,
        le=10,
        description="Number of top reranked documents to feed into Gemini.",
    )


class EvidenceItem(BaseModel):
    title: str
    url: str
    score: float
    content: str


class FactCheckResponse(BaseModel):
    claim: str
    verdict: str
    confidence: int
    explanation: str
    evidence_used: str
    citations: List[str]
    evidence: List[EvidenceItem]
    raw_result: str
    execution_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    reranker_model: str
    llm_model: str
    agent_loaded: bool


class ExampleClaim(BaseModel):
    id: str
    claim: str
    category: str
    expected_verdict: str


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serves the interactive web interface."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Agentic Fact Checker API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger API.</p>")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silence browser favicon requests."""
    return Response(status_code=204)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns the operational status and active model configurations."""
    is_cloud = bool(os.getenv("RENDER") or os.getenv("PORT") or os.getenv("LOW_MEMORY"))
    emb_model = "TF-IDF (Cloud Low-Memory Optimized)" if is_cloud else EMBEDDING_MODEL
    rerank_model = "Retriever Relevance (Cloud Zero-Overhead)" if is_cloud else RERANKER_MODEL
    return HealthResponse(
        status="operational",
        embedding_model=emb_model,
        reranker_model=rerank_model,
        llm_model=LLM_MODEL,
        agent_loaded=agent_instance is not None,
    )


@app.get("/api/examples", response_model=List[ExampleClaim], tags=["Examples"])
async def get_examples():
    """Returns curated example claims to test the pipeline."""
    return [
        ExampleClaim(
            id="ex_1",
            claim="Australia won the 2023 ICC Men's Cricket World Cup.",
            category="Sports",
            expected_verdict="SUPPORTED",
        ),
        ExampleClaim(
            id="ex_2",
            claim="The Eiffel Tower is located in Paris, France.",
            category="Geography",
            expected_verdict="SUPPORTED",
        ),
        ExampleClaim(
            id="ex_3",
            claim="India won the 2023 Men's Cricket World Cup final.",
            category="Sports",
            expected_verdict="REFUTED",
        ),
        ExampleClaim(
            id="ex_4",
            claim="Humans landed on Mars in 2020.",
            category="Science & Space",
            expected_verdict="REFUTED",
        ),
        ExampleClaim(
            id="ex_5",
            claim="NASA has officially confirmed the discovery of extraterrestrial civilizations.",
            category="Speculation",
            expected_verdict="UNCERTAIN",
        ),
    ]


@app.post("/api/verify", response_model=FactCheckResponse, tags=["Verification"])
async def verify_claim(payload: FactCheckRequest):
    """
    Executes the multi-stage agentic verification pipeline:
    1. Real-time Web Search (DDGS)
    2. Dense Semantic Retrieval (SentenceTransformers)
    3. Cross-Encoder Reranking
    4. Gemini LLM Reasoning & Synthesis
    """
    global agent_instance

    claim_text = payload.claim.strip()
    if not claim_text:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    # Lazy-load agent if startup didn't complete it
    if agent_instance is None:
        try:
            agent_instance = await asyncio.to_thread(FactCheckingAgent)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {e}")

    start_time = time.time()

    try:
        # Run agent in background thread to avoid blocking FastAPI event loop
        result = await asyncio.to_thread(
            agent_instance.run,
            claim_text,
            payload.top_k_retrieval,
            payload.top_k_rerank,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing verification pipeline: {str(e)}")

    elapsed = round(time.time() - start_time, 2)

    # Format evidence items
    evidence_items = []
    for item in result.get("structured_evidence", []):
        evidence_items.append(
            EvidenceItem(
                title=item.get("title", "Untitled Source"),
                url=item.get("url", "#"),
                score=item.get("score", 0.0),
                content=item.get("content", ""),
            )
        )

    return FactCheckResponse(
        claim=result.get("claim", claim_text),
        verdict=result.get("verdict", "UNCERTAIN"),
        confidence=result.get("confidence", 0),
        explanation=result.get("explanation", ""),
        evidence_used=result.get("evidence_used", ""),
        citations=result.get("citations", []),
        evidence=evidence_items,
        raw_result=result.get("result", ""),
        execution_time_seconds=elapsed,
    )


@app.get("/api/verify", response_model=FactCheckResponse, tags=["Verification"])
async def verify_claim_get(
    claim: str,
    top_k_retrieval: Optional[int] = TOP_K_RETRIEVAL,
    top_k_rerank: Optional[int] = TOP_K_RERANK,
):
    """
    GET method to verify a claim via URL query parameter.
    Example: /api/verify?claim=The+Eiffel+Tower+is+in+Paris
    """
    return await verify_claim(
        FactCheckRequest(
            claim=claim,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
