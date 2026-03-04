# ── Must be first — patch stderr/tqdm BEFORE any HuggingFace import ───────────
# On Windows, Flask runs without a real terminal. tqdm tries to flush sys.stderr
# when loading model weights, which raises OSError: [Errno 22] Invalid argument.
# Monkey-patching tqdm and redirecting stderr prevents this crash entirely.
import os
import sys

# Suppress all HuggingFace/tqdm progress bars at environment level
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TQDM"] = "1"

# Replace stderr with devnull if it is not a real file descriptor
_stderr_is_real = False
try:
    if sys.stderr is not None:
        sys.stderr.fileno()
        _stderr_is_real = True
except Exception:
    pass

if not _stderr_is_real:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")

# Monkey-patch tqdm globally BEFORE transformers is imported
try:
    import tqdm.std
    import tqdm.asyncio

    class _SilentTqdm:
        """No-op tqdm — never touches sys.stderr."""
        def __init__(self, *a, **kw):
            self.n = 0
            self.total = kw.get("total", 0)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __iter__(self): return iter([])
        def update(self, n=1): self.n += n
        def close(self): pass
        def set_description(self, *a, **kw): pass
        def set_postfix(self, *a, **kw): pass
        def write(self, s, **kw): pass

    tqdm.std.tqdm = _SilentTqdm
    tqdm.asyncio.tqdm = _SilentTqdm

    import tqdm
    tqdm.tqdm = _SilentTqdm
except Exception:
    pass

# ── Now safe to import everything else ────────────────────────────────────────
import traceback
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from src.config import (
    ALLOWED_EXTENSIONS,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    MAX_UPLOAD_SIZE_MB,
    SECRET_KEY,
    UPLOAD_DIR,
)
from src.embeddings import EmbeddingManager
from src.ingestion import ingest_file
from src.logger import get_logger
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

log = get_logger(__name__)


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Shared singletons — created once per process
    em = EmbeddingManager()
    vs = VectorStore()
    pipeline = RAGPipeline(vector_store=vs, embedding_manager=em)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _allowed(filename: str) -> bool:
        return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

    def _err(msg: str, status: int = 400) -> Tuple:
        log.warning("API error [%d]: %s", status, msg)
        return jsonify({"error": msg}), status

    # ── Routes ──────────────────────────────────────────────────────────────────

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "docs_in_store": vs.count})

    @app.route("/api/stats", methods=["GET"])
    def stats():
        return jsonify({
            "total_chunks": vs.count,
            "sources": vs.list_sources(),
            "embedding_model": em.model_name,
            "embedding_dim": em.dimension,
        })

    @app.route("/api/upload", methods=["POST"])
    def upload():
        if "file" not in request.files:
            return _err("No 'file' field in request.")

        file = request.files["file"]
        if not file.filename:
            return _err("Empty filename.")
        if not _allowed(file.filename):
            return _err(f"File type not allowed. Supported: {sorted(ALLOWED_EXTENSIONS)}")

        filename = secure_filename(file.filename)
        save_path = Path(UPLOAD_DIR) / filename
        file.save(str(save_path.resolve()))
        log.info("File saved: %s", save_path)

        try:
            ingested = ingest_file(save_path.resolve())
        except Exception as exc:
            log.error(traceback.format_exc())
            return _err(f"Ingestion failed: {exc}", 500)

        # Deduplication check
        force = request.args.get("force", "false").lower() == "true"
        if not force and ingested.file_hash in vs.ingested_hashes():
            return jsonify({
                "message": "File already ingested (duplicate detected).",
                "file": filename,
                "skipped": True,
            }), 200

        # Chunks are already clean from ingestion.py
        texts = [c.page_content for c in ingested.chunks if c.page_content.strip()]
        clean_chunks = [c for c in ingested.chunks if c.page_content.strip()]

        try:
            embeddings = em.encode(texts)
        except Exception as exc:
            log.error(traceback.format_exc())
            return _err(f"Embedding failed: {exc}", 500)

        try:
            inserted = vs.add_documents(clean_chunks, embeddings, ingested.file_hash)
        except Exception as exc:
            log.error(traceback.format_exc())
            return _err(f"Vector store insertion failed: {exc}", 500)

        return jsonify({
            "message": "File ingested successfully.",
            "file": filename,
            "file_type": ingested.file_type,
            "pages": ingested.num_pages,
            "chunks_inserted": inserted,
            "total_chunks_in_store": vs.count,
            "skipped": False,
        }), 201

    @app.route("/api/query", methods=["POST"])
    def query():
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()

        if not question:
            return _err("'question' field is required and cannot be empty.")

        top_k = int(data.get("top_k", 5))
        threshold = float(data.get("score_threshold", 0.25))

        if vs.count == 0:
            return _err("No documents have been ingested yet. Upload a file first.", 400)

        try:
            result = pipeline.query(question, top_k=top_k, score_threshold=threshold)
        except Exception as exc:
            log.error(traceback.format_exc())
            return _err(f"Query failed: {exc}", 500)

        return jsonify({
            "question": result.question,
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
        })

    @app.route("/api/sources", methods=["GET"])
    def list_sources():
        return jsonify({"sources": vs.list_sources()})

    @app.route("/api/sources/<path:source_name>", methods=["DELETE"])
    def delete_source(source_name: str):
        deleted = vs.delete_source(source_name)
        if deleted == 0:
            return _err(f"Source '{source_name}' not found.", 404)
        return jsonify({"deleted_chunks": deleted, "source": source_name})

    # ── Error handlers ──────────────────────────────────────────────────────────

    @app.errorhandler(413)
    def too_large(e):
        return _err(f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.", 413)

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error."}), 500

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    log.info("Starting RAG API on %s:%d (debug=%s)", FLASK_HOST, FLASK_PORT, FLASK_DEBUG)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False)