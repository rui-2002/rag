"""
src/rag_pipeline.py
───────────────────
Wires together retrieval + LLM generation.

Improvements over the original notebook:
  • Dedicated RAGPipeline class with conversation history.
  • Context window budget: trims retrieved chunks to stay under CONTEXT_CHAR_LIMIT.
  • Structured citations in every answer.
  • Graceful fallback when no relevant context is found.
  • Supports both Ollama (local) and a stub for easy swap to other LLM providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import (
    CONTEXT_CHAR_LIMIT,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.embeddings import EmbeddingManager
from src.logger import get_logger
from src.vector_store import SearchResult, VectorStore

log = get_logger(__name__)


# ── LLM wrapper ────────────────────────────────────────────────────────────────

def _build_llm():
    """
    Return an Ollama LLM instance.
    Swap this function to use a different provider (e.g. Google, OpenAI).
    """
    try:
        from langchain_community.llms import Ollama  # type: ignore
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
        log.info("LLM initialised: Ollama/%s @ %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
        return llm
    except Exception as exc:
        log.error("Failed to initialise LLM: %s", exc)
        raise


# ── Prompt construction ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions strictly based on the provided context.
If the context does not contain enough information, say so clearly — do not fabricate details.
Be concise and cite relevant sections when possible.
"""

def _build_prompt(context: str, question: str) -> str:
    return f"""{_SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float          # similarity score of top result
    context_used: str = field(repr=False, default="")


# ── Pipeline class ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.query("What is the AGR ruling?")
        print(result.answer)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self._vs = vector_store or VectorStore()
        self._em = embedding_manager or EmbeddingManager()
        self._llm = _build_llm()
        self._history: List[Dict[str, Any]] = []

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _retrieve(
        self,
        question: str,
        top_k: int,
        score_threshold: float,
    ) -> List[SearchResult]:
        q_emb = self._em.encode([question])[0]
        return self._vs.search(q_emb, top_k=top_k, score_threshold=score_threshold)

    @staticmethod
    def _build_context(results: List[SearchResult], char_limit: int = CONTEXT_CHAR_LIMIT) -> str:
        """Concatenate chunk texts, respecting the char budget."""
        parts: List[str] = []
        total = 0
        for r in results:
            snippet = r.content.strip()
            if total + len(snippet) > char_limit:
                remaining = char_limit - total
                if remaining > 100:      # only add if meaningful slice remains
                    parts.append(snippet[:remaining])
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts)

    @staticmethod
    def _build_sources(results: List[SearchResult]) -> List[Dict[str, Any]]:
        return [
            {
                "rank": r.rank,
                "source": r.metadata.get("source", "unknown"),
                "page": r.metadata.get("page", "?"),
                "score": r.similarity_score,
                "preview": r.content[:200] + ("…" if len(r.content) > 200 else ""),
            }
            for r in results
        ]

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> RAGResult:
        """
        Run the full RAG pipeline for a question.

        Args:
            question: User's natural-language question.
            top_k: Maximum number of context chunks to retrieve.
            score_threshold: Minimum similarity score to include a chunk.

        Returns:
            RAGResult with answer, sources, and confidence.
        """
        log.info("Query: '%s'", question)

        results = self._retrieve(question, top_k, score_threshold)

        if not results:
            log.warning("No relevant chunks found.")
            rag_result = RAGResult(
                question=question,
                answer="I could not find relevant information in the uploaded documents to answer this question.",
                sources=[],
                confidence=0.0,
            )
            self._history.append({"question": question, "answer": rag_result.answer})
            return rag_result

        context = self._build_context(results)
        sources = self._build_sources(results)
        confidence = results[0].similarity_score  # top result

        prompt = _build_prompt(context, question)

        log.debug(
            "Sending prompt to LLM (context=%d chars, chunks=%d)…",
            len(context), len(results),
        )
        answer: str = self._llm.invoke(prompt)

        # Append citations block
        citation_lines = [
            f"[{s['rank']}] {s['source']} — page {s['page']} (score: {s['score']:.2f})"
            for s in sources
        ]
        answer_with_citations = answer.strip() + "\n\nSources:\n" + "\n".join(citation_lines)

        rag_result = RAGResult(
            question=question,
            answer=answer_with_citations,
            sources=sources,
            confidence=confidence,
            context_used=context,
        )

        self._history.append({"question": question, "answer": answer})
        log.info("Query complete. Confidence: %.2f", confidence)
        return rag_result

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()