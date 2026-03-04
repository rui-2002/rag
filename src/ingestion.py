"""
src/ingestion.py
────────────────
Handles loading documents (PDF / TXT) and splitting them into chunks.

Improvements over the original notebook:
  • Unified loader for both file types (no separate pdf/txt branches in callers).
  • Metadata normalisation: every chunk carries source, file_type, page,
    chunk_index, char_count — useful for citations later.
  • Returns a typed dataclass instead of raw dicts so callers get IDE support.
  • Raises descriptive errors rather than silently returning empty lists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    ALLOWED_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SEPARATORS,
)
from src.logger import get_logger

log = get_logger(__name__)


# ── Public dataclass ───────────────────────────────────────────────────────────

@dataclass
class IngestedFile:
    """Result returned by ingest_file()."""
    file_path: Path
    file_type: str
    num_pages: int
    num_chunks: int
    chunks: List[Document] = field(repr=False)
    file_hash: str = ""   # SHA-256 of raw file bytes — useful for deduplication


# ── Internal helpers ───────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_pdf(path: Path) -> List[Document]:
    loader = PyMuPDFLoader(str(path))
    return loader.load()


def _load_txt(path: Path) -> List[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def _normalise_metadata(
    docs: List[Document], source_name: str, file_type: str, total_pages: int
) -> List[Document]:
    """Ensure every document has consistent metadata keys."""
    for doc in docs:
        doc.metadata["source"] = source_name
        doc.metadata["file_type"] = file_type
        doc.metadata["total_pages"] = total_pages
        # PyMuPDF already adds 'page'; TextLoader does not
        doc.metadata.setdefault("page", 0)
    return docs


def _split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(docs)

    # Add chunk-level metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["char_count"] = len(chunk.page_content)

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_file(file_path: str | Path) -> IngestedFile:
    """
    Load a single PDF or TXT file, split it into chunks, and return an
    IngestedFile object.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        IngestedFile with all chunks ready for embedding.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    log.info("Ingesting file: %s", path.name)

    if ext == ".pdf":
        raw_docs = _load_pdf(path)
        file_type = "pdf"
    else:
        raw_docs = _load_txt(path)
        file_type = "txt"

    num_pages = len(raw_docs)
    raw_docs = _normalise_metadata(raw_docs, path.name, file_type, num_pages)
    chunks = _split(raw_docs)

    fhash = _file_hash(path)

    log.info(
        "Ingested '%s': %d pages → %d chunks (hash=%s…)",
        path.name, num_pages, len(chunks), fhash[:8],
    )

    return IngestedFile(
        file_path=path,
        file_type=file_type,
        num_pages=num_pages,
        num_chunks=len(chunks),
        chunks=chunks,
        file_hash=fhash,
    )


def ingest_directory(directory: str | Path, recursive: bool = True) -> List[IngestedFile]:
    """
    Ingest all supported files in a directory.

    Args:
        directory: Path to the directory.
        recursive: Whether to search sub-directories.

    Returns:
        List of IngestedFile objects.
    """
    dir_path = Path(directory)
    pattern = "**/*" if recursive else "*"
    results: List[IngestedFile] = []

    for ext in ALLOWED_EXTENSIONS:
        for fp in dir_path.glob(f"{pattern}{ext}"):
            try:
                results.append(ingest_file(fp))
            except Exception as exc:
                log.error("Failed to ingest '%s': %s", fp.name, exc)

    log.info("Directory ingestion complete: %d files processed.", len(results))
    return results