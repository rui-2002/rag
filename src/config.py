"""
Central configuration for the RAG application.
All tuneable parameters live here — no magic numbers scattered across modules.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist at import time
for _dir in (UPLOAD_DIR, VECTOR_STORE_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── File ingestion ─────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_UPLOAD_SIZE_MB = 50

# ── Text splitting ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between consecutive chunks
SEPARATORS = ["\n\n", "\n", " ", ""]

# ── Embedding model ────────────────────────────────────────────────────────────
# Upgrade: use a larger, more accurate model vs. the original all-MiniLM-L6-v2
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = 64   # encode this many chunks at once → faster on CPU

# ── Vector store ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "rag_documents")
CHROMA_DISTANCE_METRIC = "cosine"   # ensures 1-distance = proper similarity
VECTOR_STORE_BATCH_SIZE = 500

# ── Retrieval ──────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.25      # raised from 0.0 to cut low-quality hits
CONTEXT_CHAR_LIMIT = 3000           # max chars sent to LLM per query

# ── LLM (Ollama) ───────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# ── Flask ──────────────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.getenv("PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")