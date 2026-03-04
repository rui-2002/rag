"""
src/vector_store.py
───────────────────
Manages the ChromaDB persistent vector store.

Improvements over the original notebook:
  • Document deduplication via file hash — re-uploading the same file is a no-op.
  • list_sources() and delete_source() for collection management.
  • Typed return value from search().
  • Cosine distance metric enforced at collection creation.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from langchain_core.documents import Document

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    VECTOR_STORE_BATCH_SIZE,
    VECTOR_STORE_DIR,
)
from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class SearchResult:
    """One document returned by VectorStore.search()."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int


class VectorStore:
    """
    Persistent ChromaDB-backed vector store.

    All collections use cosine distance so similarity = 1 - distance.
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str | None = None,
        batch_size: int = VECTOR_STORE_BATCH_SIZE,
    ):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory or VECTOR_STORE_DIR)
        self.batch_size = batch_size
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._init()

    # ── Private ────────────────────────────────────────────────────────────────

    def _init(self) -> None:
        os.makedirs(self.persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "RAG document embeddings",
                "hnsw:space": CHROMA_DISTANCE_METRIC,
            },
        )
        log.info(
            "VectorStore ready | collection='%s' | docs=%d",
            self.collection_name,
            self._collection.count(),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self._collection.count()

    def ingested_hashes(self) -> set[str]:
        """Return set of file hashes already in the store (for deduplication)."""
        results = self._collection.get(include=["metadatas"])
        return {
            m.get("file_hash", "")
            for m in (results["metadatas"] or [])
            if m.get("file_hash")
        }

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        file_hash: str = "",
    ) -> int:
        """
        Add documents + pre-computed embeddings to the collection.

        Args:
            documents: LangChain Document objects (with metadata).
            embeddings: numpy array, shape (len(documents), dim).
            file_hash: SHA-256 of the source file for deduplication.

        Returns:
            Number of documents actually inserted.
        """
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length.")

        ids, metas, texts, vecs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            meta = dict(doc.metadata)
            meta["doc_index"] = i
            meta["char_count"] = len(doc.page_content)
            meta["file_hash"] = file_hash
            metas.append(meta)

            texts.append(doc.page_content)
            vecs.append(emb.tolist())

        total = len(ids)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            self._collection.add(
                ids=ids[start:end],
                metadatas=metas[start:end],
                documents=texts[start:end],
                embeddings=vecs[start:end],
            )
            log.debug("Batch inserted [%d:%d]", start, end)

        log.info("Inserted %d documents. Total in store: %d", total, self.count)
        return total

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Return the top-k most similar documents.

        Args:
            query_embedding: 1-D float array from EmbeddingManager.encode().
            top_k: Maximum number of results to return.
            score_threshold: Minimum cosine similarity (0–1) to include.

        Returns:
            List of SearchResult, sorted by similarity descending.
        """
        if self.count == 0:
            log.warning("Vector store is empty — no results.")
            return []

        raw = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.count),
        )

        results: List[SearchResult] = []
        if not raw["documents"] or not raw["documents"][0]:
            return results

        for rank, (doc_id, content, meta, dist) in enumerate(
            zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            ),
            start=1,
        ):
            score = 1.0 - dist   # cosine distance → similarity
            if score >= score_threshold:
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        content=content,
                        metadata=meta,
                        similarity_score=round(score, 4),
                        rank=rank,
                    )
                )

        log.debug(
            "Query returned %d/%d results above threshold %.2f",
            len(results), top_k, score_threshold,
        )
        return results

    def list_sources(self) -> List[str]:
        """Return unique source file names stored in the collection."""
        data = self._collection.get(include=["metadatas"])
        return sorted(
            {m.get("source", "unknown") for m in (data["metadatas"] or [])}
        )

    def delete_source(self, source_name: str) -> int:
        """
        Remove all chunks belonging to a given source file.

        Returns:
            Number of chunks deleted.
        """
        data = self._collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(data["ids"], data["metadatas"] or [])
            if meta.get("source") == source_name
        ]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            log.info("Deleted %d chunks for source '%s'.", len(ids_to_delete), source_name)
        return len(ids_to_delete)

    def reset(self) -> None:
        """Drop and recreate the collection (destructive!)."""
        self._client.delete_collection(self.collection_name)
        log.warning("Collection '%s' deleted.", self.collection_name)
        self._init()