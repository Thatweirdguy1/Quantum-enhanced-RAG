"""Embedding backends.

Three interchangeable implementations behind one interface:

``ollama``   -- posts to a local Ollama daemon (default; bge-m3, 1024-d). Needs
                no torch install, which keeps the dev environment light.
``sbert``    -- sentence-transformers, the backend named in the synopsis. Used
                to confirm that results are not an artefact of one embedder.
``hashing``  -- deterministic character-n-gram hashing. No network, no weights;
                exists so the test suite can run anywhere. Never used for
                reported numbers.

All backends return L2-normalised float32 vectors, so a dot product *is* cosine
similarity and the amplitude-encoding step in :mod:`qrag.kernel` is valid
without further rescaling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import requests

from .config import CACHE_DIR, EmbedConfig
from .security import safe_tag


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder:
    """Base class. Subclasses implement :meth:`_encode_batch`."""

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.dim = cfg.dim

    @property
    def signature(self) -> str:
        return f"{self.cfg.backend}-{self.cfg.model}"

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        bs = self.cfg.batch_size
        for start in range(0, len(texts), bs):
            chunk = [t[: self.cfg.max_chars] for t in texts[start : start + bs]]
            out[start : start + len(chunk)] = self._encode_batch(chunk)
            if show_progress and (start // bs) % 20 == 0:
                pct = 100.0 * min(start + bs, len(texts)) / max(len(texts), 1)
                print(f"    embedding {min(start + bs, len(texts))}/{len(texts)}"
                      f" ({pct:.0f}%)", flush=True)
        return l2_normalise(out)

    def encode_cached(self, texts: list[str], tag: str,
                      show_progress: bool = True) -> np.ndarray:
        """Encode with an on-disk cache keyed by backend, tag and text content.

        Embedding 5k abstracts through a local model takes minutes; doing it once
        per corpus rather than once per experiment is the difference between a
        usable iteration loop and an unusable one.
        """
        digest = hashlib.sha256(
            (self.signature + "|" + tag + "|" + str(len(texts))
             + "|" + "".join(t[:64] for t in texts[:200])).encode()
        ).hexdigest()[:16]
        # tag is interpolated into a filename, so it is sanitised to a single path
        # component. The digest is computed over the raw tag so that sanitising two
        # different tags to the same string cannot collide their cache entries.
        path = CACHE_DIR / f"emb-{safe_tag(tag)}-{safe_tag(self.signature)}-{digest}.npy"
        if path.exists():
            cached = np.load(path)
            if cached.shape == (len(texts), self.dim):
                print(f"  [cache hit] {path.name}")
                return cached
        print(f"  [cache miss] encoding {len(texts)} texts with {self.signature}")
        vectors = self.encode(texts, show_progress=show_progress)
        np.save(path, vectors)
        return vectors


class OllamaEmbedder(Embedder):
    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        self.url = cfg.ollama_url.rstrip("/") + "/api/embed"
        self.dim = self._probe_dim()

    def _probe_dim(self) -> int:
        vec = self._raw(["dimension probe"])
        return len(vec[0])

    def _raw(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            self.url,
            json={"model": self.cfg.model, "input": texts},
            timeout=300,
        )
        resp.raise_for_status()
        payload = resp.json()
        vectors = payload.get("embeddings")
        if vectors is None:  # older daemons return a single "embedding"
            vectors = [payload["embedding"]]
        return vectors

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._raw(texts), dtype=np.float32)


class SbertEmbedder(Embedder):
    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(cfg.model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=False)


class HashingEmbedder(Embedder):
    """Deterministic character-trigram hashing. Offline fallback only."""

    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        self.dim = 512

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            for i in range(len(lowered) - 2):
                gram = lowered[i : i + 3]
                h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
                out[row, h % self.dim] += 1.0
        return np.log1p(out)


def build_embedder(cfg: EmbedConfig) -> Embedder:
    backends = {
        "ollama": OllamaEmbedder,
        "sbert": SbertEmbedder,
        "hashing": HashingEmbedder,
    }
    if cfg.backend not in backends:
        raise ValueError(f"unknown embed backend {cfg.backend!r}; "
                         f"choose from {sorted(backends)}")
    return backends[cfg.backend](cfg)
