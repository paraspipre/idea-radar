"""Local text embeddings (Anthropic has no embeddings API; nothing here leaves the machine)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


def normalize(matrix: NDArray[Any]) -> Matrix:
    m = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return cast(Matrix, (m / norms).astype(np.float32))


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> Matrix:
        """Return an (n, dim) float32 matrix of L2-normalized rows."""
        ...


class FastEmbedder:
    """fastembed (ONNX, CPU). The model is downloaded once (~130 MB) and cached."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.name = model_name
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
        vectors = list(self._model.embed(list(texts)))
        return normalize(np.vstack(vectors))


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model download)."""

    name = "hash-test"

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                digest = hashlib.md5(word.strip(".,:;!?").encode()).digest()
                out[row, digest[0] % self._dim] += 1.0
        return normalize(out)


def to_blob(vector: NDArray[Any]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> Matrix:
    return np.frombuffer(blob, dtype=np.float32).copy()
