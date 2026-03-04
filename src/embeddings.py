"""
src/embeddings.py
─────────────────
Wraps SentenceTransformer with batching, caching, and lazy loading.

Improvements over the original notebook:
  • Lazy model load: model is loaded once on first call, not at import time.
  • Configurable batch size: encodes large corpora faster without OOM.
  • encode() returns a numpy array — same as before but typed.
  • Thread-safe singleton pattern (one model per process).
  • tqdm/stderr disabled — prevents [Errno 22] on Windows when running under Flask.
"""

from __future__ import annotations
import sys, os
if not hasattr(sys.stderr, 'fileno') or sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
import threading
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from src.logger import get_logger

log = get_logger(__name__)


def _silence_tqdm() -> None:
    """
    Redirect stderr to devnull before loading the model.
    tqdm tries to flush sys.stderr during model weight loading, which raises
    OSError: [Errno 22] Invalid argument on Windows when running under Flask
    (no real terminal attached). This prevents that crash.
    """
    # In PowerShell jobs/background runs on Windows, stderr can have a fileno
    # but still fail later on flush with OSError(22). Always redirect to a
    # safe sink in non-interactive execution to avoid tqdm crashes.
    if os.name == "nt" and not sys.stderr.isatty():
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")
        return

    if sys.stderr is None or not hasattr(sys.stderr, "fileno"):
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")
        return

    try:
        sys.stderr.fileno()
    except Exception:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")


def _patch_tqdm() -> None:
    """Hard-disable tqdm constructors to avoid stderr flush crashes."""
    try:
        import tqdm
        import tqdm.auto
        import tqdm.asyncio
        import tqdm.std

        class _SilentTqdm:
            def __init__(self, *args, **kwargs):
                self.n = 0
                self.total = kwargs.get("total", 0)
                if len(args) == 0:
                    self._iterable = []
                elif len(args) == 1:
                    self._iterable = args[0]
                else:
                    self._iterable = range(*args)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def update(self, n=1):
                self.n += n

            def __iter__(self):
                try:
                    return iter(self._iterable)
                except TypeError:
                    return iter(range(int(self._iterable)))

            def close(self):
                return None

            def set_description(self, *args, **kwargs):
                return None

            def set_postfix(self, *args, **kwargs):
                return None

        tqdm.tqdm = _SilentTqdm
        tqdm.auto.tqdm = _SilentTqdm
        tqdm.asyncio.tqdm = _SilentTqdm
        tqdm.std.tqdm = _SilentTqdm
    except Exception:
        # If tqdm isn't available, nothing else to patch.
        pass


class EmbeddingManager:
    """
    Singleton-safe wrapper around SentenceTransformer.

    Usage:
        em = EmbeddingManager()
        vecs = em.encode(["text1", "text2"])
    """

    _instance: "EmbeddingManager | None" = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = EMBEDDING_MODEL):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._model_name = model_name
                instance._model: SentenceTransformer | None = None
                cls._instance = instance
            return cls._instance

    # ── Private ────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is None:
            log.info("Loading embedding model: %s ...", self._model_name)

            # Disable HuggingFace/tqdm progress bars — they crash on Windows
            # under Flask because there is no real stderr/terminal attached.
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

            import transformers
            transformers.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()

            try:
                import datasets
                datasets.disable_progress_bars()
            except Exception:
                # `datasets` is optional for this app.
                pass

            _silence_tqdm()
            _patch_tqdm()

            self._model = SentenceTransformer(
                self._model_name,
                device="cpu",
            )
            log.info(
                "Model loaded. Embedding dimension: %d",
                self._model.get_sentence_embedding_dimension(),
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.
            batch_size: How many texts to encode per forward pass.
            show_progress: Show tqdm progress bar (safe only in terminals).

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._ensure_loaded()
        log.debug("Encoding %d texts (batch_size=%d)...", len(texts), batch_size)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,   # always off — safe under Flask/Windows
            convert_to_numpy=True,
            normalize_embeddings=True,  # pre-normalise for cosine similarity
        )

        log.debug("Encoding complete. Shape: %s", embeddings.shape)
        return embeddings
