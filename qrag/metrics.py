"""Information-retrieval metrics and paired significance testing.

Metric definitions follow TREC/BEIR conventions so numbers are comparable with
published SciFact results. All three metrics are computed per query and returned
alongside the mean, because the paired bootstrap needs the per-query vector --
and because a mean with no dispersion is not a reportable result on 300 queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int | None = None) -> float:
    horizon = ranked if k is None else ranked[:k]
    for i, doc in enumerate(horizon, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """nDCG with the standard 2^rel - 1 gain and log2(rank + 1) discount."""
    gains = [(2.0 ** qrels.get(doc, 0)) - 1.0 for doc in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(((2.0**r) - 1.0) / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


@dataclass
class EvalReport:
    system: str
    k_values: tuple
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: list[float] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def add(self, query_id: str, ranked: list[str], qrels_q: dict[str, int],
            latency_ms: float | None = None) -> None:
        relevant = {d for d, s in qrels_q.items() if s > 0}
        row: dict[str, float] = {}
        for k in self.k_values:
            row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
            row[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, qrels_q, k)
        row["mrr"] = reciprocal_rank(ranked, relevant)
        row["mrr@10"] = reciprocal_rank(ranked, relevant, 10)
        self.per_query[query_id] = row
        if latency_ms is not None:
            self.latency_ms.append(latency_ms)

    def vector(self, metric: str) -> np.ndarray:
        return np.array([r[metric] for r in self.per_query.values()])

    def mean(self, metric: str) -> float:
        return float(np.mean(self.vector(metric))) if self.per_query else 0.0

    @property
    def metrics(self) -> list[str]:
        return sorted(next(iter(self.per_query.values())).keys()) if self.per_query else []

    def means(self) -> dict[str, float]:
        out = {m: self.mean(m) for m in self.metrics}
        if self.latency_ms:
            out["latency_ms_mean"] = float(np.mean(self.latency_ms))
            out["latency_ms_p95"] = float(np.percentile(self.latency_ms, 95))
        return out

    def table(self, keys: list[str] | None = None) -> str:
        keys = keys or ["recall@10", "ndcg@10", "mrr@10"]
        means = self.means()
        parts = [f"{k}={means.get(k, float('nan')):.4f}" for k in keys]
        return f"{self.system:<22} " + "  ".join(parts)


# ------------------------------------------------------------------ significance
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_samples: int = 2000,
                     seed: int = 20260720) -> dict:
    """Paired bootstrap over queries for the difference in means (b - a).

    The paired form matters: the same queries run through both systems, so
    per-query difficulty is a shared nuisance factor that pairing removes. On 300
    queries an unpaired test would need a far larger effect to reach the same
    confidence.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired test needs equal-length vectors")
    n = len(a)
    observed = float(np.mean(b) - np.mean(a))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_samples, n))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p: proportion of resamples on the opposite side of zero.
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {
        "delta": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": min(1.0, p),
        "significant": bool(lo > 0 or hi < 0),
        "n_queries": n,
        "n_samples": n_samples,
    }


def compare_reports(baseline: EvalReport, system: EvalReport,
                    metrics: list[str] | None = None,
                    n_samples: int = 2000, seed: int = 20260720) -> dict:
    """Paired comparison on every metric, aligned by query id."""
    metrics = metrics or ["recall@10", "ndcg@10", "mrr@10", "recall@5", "ndcg@5"]
    shared = [q for q in baseline.per_query if q in system.per_query]
    out = {}
    for metric in metrics:
        a = np.array([baseline.per_query[q][metric] for q in shared])
        b = np.array([system.per_query[q][metric] for q in shared])
        out[metric] = paired_bootstrap(a, b, n_samples, seed)
        out[metric]["baseline_mean"] = float(np.mean(a))
        out[metric]["system_mean"] = float(np.mean(b))
    return out


def format_comparison(comparison: dict, baseline_name: str = "baseline",
                      system_name: str = "Q-RAG") -> str:
    lines = [
        f"{'metric':<14}{baseline_name:>12}{system_name:>12}{'delta':>10}"
        f"{'95% CI':>22}{'p':>9}  sig",
        "-" * 82,
    ]
    for metric, r in comparison.items():
        ci = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
        lines.append(
            f"{metric:<14}{r['baseline_mean']:>12.4f}{r['system_mean']:>12.4f}"
            f"{r['delta']:>+10.4f}{ci:>22}{r['p_value']:>9.4f}"
            f"  {'yes' if r['significant'] else 'no'}"
        )
    return "\n".join(lines)
