"""Hybrid fusion of lexical, dense and quantum-kernel scores.

BM25 is unbounded, cosine lives in [-1, 1], and the fidelity kernel in [0, 1].
Mixing them without normalisation would let BM25's scale silently dominate the
weights, so every leg is rescaled per query before the weighted sum. Rescaling
is per-query rather than global because BM25's dynamic range varies enormously
between a two-word query and a full sentence claim.
"""

from __future__ import annotations

import numpy as np


def minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def zscore(scores: np.ndarray) -> np.ndarray:
    mu, sd = float(np.mean(scores)), float(np.std(scores))
    return np.zeros_like(scores) if sd < 1e-12 else (scores - mu) / sd


NORMALISERS = {"minmax": minmax, "zscore": zscore, "none": lambda s: s}


def fuse(components: dict[str, np.ndarray], weights: dict[str, float],
         normalise: str = "minmax") -> np.ndarray:
    """Weighted sum of named score vectors, each normalised independently."""
    fn = NORMALISERS[normalise]
    total = None
    for name, vec in components.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        contribution = w * fn(np.asarray(vec, dtype=np.float64))
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("all fusion weights are zero")
    return total


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF: rank-based fusion that needs no score normalisation at all.

    Kept as a comparator for the weighted scheme, since RRF is the standard
    strong baseline for hybrid retrieval and a weighted sum that cannot beat it
    is not worth the tuning effort it costs.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
