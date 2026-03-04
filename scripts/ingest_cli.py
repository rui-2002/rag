"""
scripts/ingest_cli.py
─────────────────────
Command-line tool to ingest a file or directory into the vector store.

Usage:
    python scripts/ingest_cli.py --file path/to/file.pdf
    python scripts/ingest_cli.py --dir path/to/pdf_folder
    python scripts/ingest_cli.py --dir path/to/folder --reset
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import EmbeddingManager
from src.ingestion import ingest_directory, ingest_file
from src.logger import get_logger
from src.vector_store import VectorStore

log = get_logger("ingest_cli")


def ingest_one(path: Path, vs: VectorStore, em: EmbeddingManager) -> None:
    log.info("Processing: %s", path.name)
    ingested = ingest_file(path)

    if ingested.file_hash in vs.ingested_hashes():
        log.info("Skipping duplicate: %s", path.name)
        return

    embeddings = em.encode([c.page_content for c in ingested.chunks])
    vs.add_documents(ingested.chunks, embeddings, ingested.file_hash)
    log.info("Done: %s (%d chunks)", path.name, ingested.num_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector store.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single PDF or TXT file.")
    group.add_argument("--dir", help="Path to a directory of files.")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store first.")
    args = parser.parse_args()

    vs = VectorStore()
    em = EmbeddingManager()

    if args.reset:
        log.warning("Resetting vector store…")
        vs.reset()

    if args.file:
        ingest_one(Path(args.file), vs, em)
    else:
        results = ingest_directory(args.dir)
        for ingested in results:
            if ingested.file_hash in vs.ingested_hashes():
                log.info("Skipping duplicate: %s", ingested.file_path.name)
                continue
            embeddings = em.encode([c.page_content for c in ingested.chunks])
            vs.add_documents(ingested.chunks, embeddings, ingested.file_hash)

    log.info("Ingestion complete. Total chunks in store: %d", vs.count)


if __name__ == "__main__":
    main()