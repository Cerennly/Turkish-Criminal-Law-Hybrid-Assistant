"""
FastAPI application: POST /ask for legal Q&A with source citations.
Serves SaaS-style web UI at / for public use.
Logs all queries; refuses answer when retrieval score is below threshold.
"""
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import API_TITLE, API_VERSION, LOG_DIR, SIMILARITY_SCORE_THRESHOLD
from app.schemas import AskRequest, AskResponse, SourceRef
from app.rag.chain import run_rag

# Configure query logging to file
LOG_FILE = LOG_DIR / "queries.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title=API_TITLE, version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Web UI (SaaS-style): serve static files and index at /
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    """Serve the Legal AI web UI so anyone can try via link."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Legal AI API", "docs": "/docs", "ask": "POST /ask"}


@app.get("/health")
def health():
    """Health check for deployment."""
    return {"status": "ok", "service": API_TITLE}


def _run_ask(request: AskRequest):
    """Shared logic for /ask and /api/ask."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    logger.info("Query: %s", question)
    try:
        answer, sources = run_rag(question, refuse_below_threshold=True)
        return AskResponse(
            answer=answer,
            sources=[SourceRef(document=s["document"], page=s["page"]) for s in sources],
        )
    except FileNotFoundError as e:
        logger.exception("Vector store not found. Run ingestion first.")
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded. Run the ingestion script first (see README).",
        ) from e
    except Exception as e:
        logger.exception("RAG error: %s", e)
        raise HTTPException(status_code=500, detail="An error occurred while processing your question.") from e


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a legal question strictly from ingested documents. Returns answer and sources."""
    return _run_ask(request)


@app.post("/api/ask", response_model=AskResponse)
def api_ask(request: AskRequest):
    """Same as /ask; used by web UI and Vercel proxy."""
    return _run_ask(request)
