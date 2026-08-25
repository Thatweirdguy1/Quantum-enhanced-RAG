"""Retrieval pipelines.

:class:`BaselineRAG` and :class:`QRAG` expose the same ``retrieve`` signature, so
the comparison is a configuration change rather than a separate code path. Every
quantum component on :class:`QRAG` is independently switchable, which is what
makes a component-wise ablation possible instead of an all-or-nothing claim.

Q-RAG stage order
-----------------
1. Classical pre-filter -- BM25 + dense cosine fused, keep the top ``S``
   candidates. Necessary because neither the fidelity kernel nor any quantum
   component can be applied corpus-wide at reasonable cost.
2. Kernel rescoring -- phase-modulated fidelity over the shortlist, optionally
   with interference across pseudo-relevance-feedback query views.
3. Grover narrowing -- amplitude amplification against a threshold oracle,
   recorded for query-complexity accounting (see :mod:`qrag.grover` on why this
   is not a latency claim).
4. QAOA reranking -- the top ``n`` survivors become a QUBO; the selected ``k``
   form the context set.
5. Tail -- remaining shortlist appended by fused score so that metrics at
   ``k > qaoa_k`` remain well defined.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .data import Dataset
from .fusion import fuse, minmax
from .grover import GroverShortlister
from .index import BM25Index, DenseIndex
from .kernel import PhaseKernel
from .qaoa import QUBO, rerank_exact, rerank_qaoa, selection_redundancy
from .security import validate_query, validate_top_k


@dataclass
class RetrievalResult:
    query_id: str
    ranked: list[str]
    scores: dict[str, float]
    latency_ms: float
    trace: dict = field(default_factory=dict)

    def top(self, k: int) -> list[str]:
        return self.ranked[:k]


class BaselineRAG:
    """Classical hybrid retrieval: BM25 + dense cosine, weighted fusion."""

    name = "classical-baseline"

    def __init__(self, dataset: Dataset, dense: DenseIndex, bm25: BM25Index,
                 query_vectors: dict[str, np.ndarray], cfg: Config):
        self.ds = dataset
        self.dense = dense
        self.bm25 = bm25
        self.qvecs = query_vectors
        self.cfg = cfg
        self.doc_ids = dataset.doc_ids
        self.position = {d: i for i, d in enumerate(self.doc_ids)}

    def _candidate_pool(self, query_text: str, qvec: np.ndarray, size: int):
        """Union of the dense and lexical top lists, with both score legs."""
        dense_ids, _ = self.dense.search(qvec, size)
        bm25_all = self.bm25.scores(query_text)
        lex_idx = np.argpartition(-bm25_all, min(size, len(bm25_all) - 1))[:size]
        pool = list(dict.fromkeys(dense_ids + [self.doc_ids[i] for i in lex_idx]))
        idx = np.array([self.position[d] for d in pool])
        return pool, idx, self.dense.vectors[idx] @ qvec, bm25_all[idx]

    def retrieve(self, query_id: str, query_text: str, top_k: int = 20) -> RetrievalResult:
        t0 = time.perf_counter()
        # Both pipelines validate here rather than at the caller, because this is
        # the library's entry point: query_text reaches the BM25 tokeniser and
        # top_k sizes every array downstream.
        query_text = validate_query(query_text)
        top_k = validate_top_k(top_k)
        qvec = self.qvecs[query_id]
        pool_size = max(top_k * 5, self.cfg.quantum.grover_shortlist)
        pool, _idx, cos, bm = self._candidate_pool(query_text, qvec, pool_size)

        fused = fuse(
            {"bm25": bm, "cosine": cos},
            {"bm25": self.cfg.fusion.w_bm25,
             # The kernel weight is redistributed onto cosine so the baseline is
             # a properly tuned hybrid rather than a deliberately weak strawman.
             "cosine": self.cfg.fusion.w_cosine + self.cfg.fusion.w_kernel},
            self.cfg.fusion.normalise,
        )
        order = np.argsort(-fused)
        ranked = [pool[i] for i in order]
        latency = (time.perf_counter() - t0) * 1e3
        return RetrievalResult(
            query_id=query_id,
            ranked=ranked[:top_k],
            scores={pool[i]: float(fused[i]) for i in order[:top_k]},
            latency_ms=latency,
            trace={"pool_size": len(pool), "stage": "bm25+cosine"},
        )


class QRAG:
    """Quantum-inspired pipeline. Each stage can be switched off independently."""

    def __init__(self, dataset: Dataset, dense: DenseIndex, bm25: BM25Index,
                 query_vectors: dict[str, np.ndarray], kernel: PhaseKernel,
                 cfg: Config, *, use_kernel: bool = True,
                 use_interference: bool = True, use_grover: bool = True,
                 use_qaoa: bool = True, label: str | None = None):
        self.ds = dataset
        self.dense = dense
        self.bm25 = bm25
        self.qvecs = query_vectors
        self.kernel = kernel
        self.cfg = cfg
        self.use_kernel = use_kernel
        self.use_interference = use_interference
        self.use_grover = use_grover
        self.use_qaoa = use_qaoa
        self.doc_ids = dataset.doc_ids
        self.position = {d: i for i, d in enumerate(self.doc_ids)}
        self.grover = GroverShortlister(cfg.quantum.grover_threshold_quantile,
                                        cfg.quantum.backend)
        self._baseline = BaselineRAG(dataset, dense, bm25, query_vectors, cfg)
        self.name = label or self._auto_label()

    def _auto_label(self) -> str:
        flags = [n for n, on in (("kernel", self.use_kernel),
                                 ("interf", self.use_interference),
                                 ("grover", self.use_grover),
                                 ("qaoa", self.use_qaoa)) if on]
        return "qrag[" + "+".join(flags) + "]" if flags else "qrag[none]"

    # ------------------------------------------------------------------ stages
    def _query_views(self, qvec: np.ndarray, pool_idx: np.ndarray,
                     cos: np.ndarray) -> np.ndarray:
        """Literal query plus top pseudo-relevance-feedback expansions."""
        n_views = max(1, self.cfg.kernel.n_subqueries)
        views = [qvec]
        if n_views > 1:
            for i in np.argsort(-cos)[: n_views - 1]:
                views.append(self.dense.vectors[pool_idx[i]])
        return np.stack(views)

    def retrieve(self, query_id: str, query_text: str, top_k: int = 20) -> RetrievalResult:
        t0 = time.perf_counter()
        query_text = validate_query(query_text)
        top_k = validate_top_k(top_k)
        qcfg = self.cfg.quantum
        qvec = self.qvecs[query_id]
        trace: dict = {}

        # --- stage 1: classical pre-filter
        pool_size = max(top_k * 5, qcfg.grover_shortlist)
        pool, idx, cos, bm = self._baseline._candidate_pool(query_text, qvec, pool_size)
        pool_vecs = self.dense.vectors[idx]

        # --- stage 2: quantum kernel rescoring
        if self.use_kernel:
            if self.use_interference and self.cfg.kernel.n_subqueries > 1:
                views = self._query_views(qvec, idx, cos)
                kern = self.kernel.interference_score(views, pool_vecs)
                trace["kernel_mode"] = f"interference-{len(views)}views"
                trace["kernel_additive_ref"] = float(
                    np.mean(self.kernel.additive_score(views, pool_vecs)))
            else:
                kern = self.kernel.score(qvec, pool_vecs)
                trace["kernel_mode"] = "fidelity"
            weights = {"bm25": self.cfg.fusion.w_bm25,
                       "cosine": self.cfg.fusion.w_cosine,
                       "kernel": self.cfg.fusion.w_kernel}
            fused = fuse({"bm25": bm, "cosine": cos, "kernel": kern},
                         weights, self.cfg.fusion.normalise)
        else:
            kern = None
            fused = fuse({"bm25": bm, "cosine": cos},
                         {"bm25": self.cfg.fusion.w_bm25,
                          "cosine": self.cfg.fusion.w_cosine
                          + self.cfg.fusion.w_kernel},
                         self.cfg.fusion.normalise)

        # --- stage 3: Grover narrowing over the shortlist
        shortlist_n = min(qcfg.grover_shortlist, len(pool))
        short_order = np.argsort(-fused)[:shortlist_n]
        if self.use_grover:
            gres = self.grover.run(fused[short_order])
            trace["grover"] = {
                "n_qubits": gres.n_qubits,
                "n_candidates": gres.n_candidates,
                "n_marked": gres.n_marked,
                "oracle_queries": gres.oracle_queries,
                "classical_expected_queries": gres.classical_expected_queries,
                "query_reduction_factor": gres.query_reduction_factor,
                "success_probability": gres.success_probability,
                "wall_clock_ms": gres.wall_clock_s * 1e3,
                "simulation_overhead": gres.simulation_overhead,
            }
            # Amplification concentrates probability on the marked (above
            # threshold) set; that ordering promotes the marked block while
            # leaving relative order inside it driven by the fused score.
            amp_rank = [i for i in gres.ranked_indices if i < shortlist_n]
            short_order = short_order[amp_rank]

        # --- stage 4: QAOA reranking of the surviving head
        cand_n = min(qcfg.qaoa_candidates, len(short_order))
        if self.use_qaoa and cand_n >= qcfg.qaoa_k + 1:
            head = short_order[:cand_n]
            rel = minmax(fused[head])
            hv = pool_vecs[head]
            sim = np.clip(hv @ hv.T, 0.0, 1.0)
            np.fill_diagonal(sim, 0.0)
            qubo = QUBO(rel, sim, qcfg.qaoa_k, qcfg.qaoa_redundancy_lambda,
                        qcfg.qaoa_cardinality_mu)
            exact = rerank_exact(qubo)
            qres = rerank_qaoa(qubo, qcfg.qaoa_layers, qcfg.qaoa_optimiser_iters,
                               qcfg.seed, qcfg.backend, exact.objective)
            selected = [int(head[i]) for i in qres.selected]
            selected.sort(key=lambda i: -fused[i])
            tail = [int(i) for i in short_order if int(i) not in set(selected)]
            final_idx = selected + tail
            trace["qaoa"] = {
                "n_qubits": qres.n_qubits,
                "layers": qres.layers,
                "objective": qres.objective,
                "exact_objective": exact.objective,
                # Affine-invariant quality in [0, 1], 1 == exact optimum. Not
                # achieved/optimal: that form inverts when the objective is
                # positive, which happens on queries whose candidates are weakly
                # relevant and mutually similar. See qaoa.solution_quality.
                "solution_quality": qres.approximation_ratio,
                "is_optimal": qres.extras["is_optimal"],
                "objective_gap": qres.extras["objective_gap"],
                "worst_feasible_objective": qres.extras["worst_feasible_objective"],
                "feasible_probability": qres.feasible_probability,
                "optimiser_calls": qres.optimiser_iters,
                "wall_clock_ms": qres.wall_clock_s * 1e3,
                "redundancy_qaoa": selection_redundancy(qubo, qres.selected),
                "redundancy_topk": selection_redundancy(
                    qubo, sorted(np.argsort(-rel)[: qcfg.qaoa_k].tolist())),
            }
        else:
            final_idx = [int(i) for i in short_order]

        ranked = [pool[i] for i in final_idx]
        latency = (time.perf_counter() - t0) * 1e3
        trace["pool_size"] = len(pool)
        trace["shortlist"] = shortlist_n
        return RetrievalResult(
            query_id=query_id,
            ranked=ranked[:top_k],
            scores={pool[i]: float(fused[i]) for i in final_idx[:top_k]},
            latency_ms=latency,
            trace=trace,
        )


def build_indexes(dataset: Dataset, doc_vectors: np.ndarray,
                  use_faiss: bool = True) -> tuple[DenseIndex, BM25Index]:
    dense = DenseIndex(doc_vectors, dataset.doc_ids, use_faiss=use_faiss)
    bm25 = BM25Index([d.content for d in dataset.documents], dataset.doc_ids)
    return dense, bm25
