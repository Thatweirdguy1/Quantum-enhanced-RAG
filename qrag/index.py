"""Retrieval indexes: dense vector search and lexical BM25.

At SciFact scale (5k documents) an exact inner-product scan is both faster and
more accurate than any approximate structure, so ``DenseIndex`` defaults to
FAISS ``IndexFlatIP``. The ANN path is kept behind the same interface because
the O(N) argument in the project's motivation only bites at corpus sizes where
exact search stops being viable, and we want to be able to demonstrate that
crossover rather than assert it.
"""

from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class DenseIndex:
    """Inner-product search over L2-normalised embeddings (== cosine)."""

    def __init__(self, vectors: np.ndarray, doc_ids: list[str], use_faiss: bool = True):
        if vectors.shape[0] != len(doc_ids):
            raise ValueError("vectors and doc_ids disagree on length")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.doc_ids = doc_ids
        self.backend = "numpy"
        self._index = None
        if use_faiss:
            try:
                import faiss

                self._index = faiss.IndexFlatIP(self.vectors.shape[1])
                self._index.add(self.vectors)
                self.backend = "faiss-IndexFlatIP"
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"  [DenseIndex] FAISS unavailable ({exc}); using numpy scan")

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[list[str], np.ndarray]:
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        if self._index is not None:
            scores, idx = self._index.search(q, top_k)
            scores, idx = scores[0], idx[0]
        else:
            sims = self.vectors @ q[0]
            idx = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
            idx = idx[np.argsort(-sims[idx])]
            scores = sims[idx]
        return [self.doc_ids[i] for i in idx], scores

    def scores_for(self, query_vec: np.ndarray, doc_indices: np.ndarray) -> np.ndarray:
        return self.vectors[doc_indices] @ query_vec


class BM25Index:
    """Okapi BM25 over the same document set as the dense index."""

    def __init__(self, corpus_texts: list[str], doc_ids: list[str]):
        from rank_bm25 import BM25Okapi

        self.doc_ids = doc_ids
        self.position = {d: i for i, d in enumerate(doc_ids)}
        self._bm25 = BM25Okapi([tokenise(t) for t in corpus_texts])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenise(query)), dtype=np.float32)

    def search(self, query: str, top_k: int) -> tuple[list[str], np.ndarray]:
        s = self.scores(query)
        idx = np.argpartition(-s, min(top_k, len(s) - 1))[:top_k]
        idx = idx[np.argsort(-s[idx])]
        return [self.doc_ids[i] for i in idx], s[idx]
