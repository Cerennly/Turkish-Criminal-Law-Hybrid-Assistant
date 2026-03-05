"""
Application configuration for Legal AI system.
Centralizes all configurable parameters for ingestion, retrieval, and generation.
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"

# Ingestion
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "BAAI/bge-m3"  # Turkish-capable multilingual embedding model

# Retrieval
RETRIEVAL_K = 4
# Minimum relevance score (0–1) to consider retrieval confident; refuse answer if best score below this
SIMILARITY_SCORE_THRESHOLD = 0.5

# Generation (Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # Or mistral, qwen2, etc. - ensure model supports Turkish
TEMPERATURE = 0.1
TOP_P = 0.9

# API
API_TITLE = "Turkish Criminal Law AI"
API_VERSION = "1.0.0"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
