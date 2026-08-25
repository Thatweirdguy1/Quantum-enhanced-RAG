r"""Quantum-inspired similarity kernels.

Two constructions live here, together with the measurements that decide what may
be claimed about them. The headline is not a speedup or an accuracy win; it is
that **the training signal, not the circuit, decides whether a phase kernel does
anything at all**, and that even when it does, it earns its place as a
complementary signal rather than a better one.

1. Global fidelity (:class:`GlobalFidelityKernel`)
--------------------------------------------------
Encode an L2-normalised embedding as an amplitude vector, apply a learned
diagonal unitary to the document, and measure the overlap:

.. math::
    K_\theta(q, d) = |\langle \psi_q | U(\theta) | \psi_d \rangle|^2
                   = \Big| \sum_i q_i d_i e^{i\theta_i} \Big|^2

At :math:`\theta = 0` this is :math:`\cos^2\angle(q,d)`, a monotone transform of
cosine over the non-negative range, hence rank-identical to the dense baseline.
Measured directly: :meth:`BlockFidelityKernel.cos2_score` and cosine give
identical held-out top-1 and MRR to four decimal places.

*The negative result, and its correction.* Trained against **uniformly sampled**
negatives, this kernel collapses to :math:`\theta \approx 0`: mean
:math:`|\theta|` fell 0.010 to 0.002 over 40 epochs while Kendall's tau against
cosine rose to 0.99999. The reason is that cosine already separates a positive
from a random negative about 95% of the time, so there is no residual error for
the phases to reduce, and any rotation costs the positive some of its own score
through destructive interference. We first read this as a structural property of
the construction. **That was wrong**, and the same code refutes it: retrained
against *mined hard* negatives the collapse does not occur -- mean
:math:`|\theta|` reaches 0.92 and tau against cosine falls to 0.45. The negative
result is therefore about the supervision, not the circuit, and it is reported
that way. See :func:`mine_hard_negatives`.

2. Block fidelity (:class:`BlockFidelityKernel`) -- the pipeline's kernel
------------------------------------------------------------------------
Measure *local* overlaps on blocks of the amplitude vector and learn how to
weight them, rather than measuring one global overlap:

.. math::
    K(q, d) = \sum_{g} w_g \Big| \sum_{i \in g} q_i d_i e^{i\theta_i} \Big|^2

This is the projected-kernel idea from quantum machine learning: local
observables on subsystems instead of global state fidelity. Two consequences.

* Even at :math:`\theta = 0` it equals :math:`\sum_g S_g^2` where
  :math:`S_g = \sum_{i \in g} q_i d_i`, and by Cauchy-Schwarz that is *not* a
  monotone function of :math:`(\sum_g S_g)^2 = \cos^2`. Cosine-equivalence is
  broken structurally, before any training: the untrained, zero-phase kernel
  already sits at tau 0.40 against cosine. Unlike the global case this claim did
  survive measurement.
* Phases vary *within* a block, so they modulate how dimensions inside a block
  interfere instead of applying one global rotation. A phase shared across a
  whole block factors out of the modulus and does nothing, which is why
  :attr:`BlockFidelityKernel.theta` is per-dimension while the weights are
  per-block.

3. What the kernel is actually worth
------------------------------------
Reordering is not improving, and the two must be gated separately -- a random
projection reorders too. On 137 held-out pairs, each positive scored against the
15 hardest mined negatives for its query:

================================  ======  ======
scorer                             top-1     MRR
================================  ======  ======
cosine                            0.6277  0.6881
cosine squared                    0.6277  0.6881
global fidelity, trained          0.6204  0.6986
block fidelity, trained           0.5985  0.6962
block fidelity, :math:`\theta=0`  0.4964  0.5921
cosine + 0.25 x block (fused)     0.6423  0.7034
================================  ======  ======

Read standalone, the block kernel **loses 2.9 top-1 points to cosine** while
gaining 0.8 MRR points: it lifts relevant documents out of mid-list but displaces
the ones cosine already ranked first. That is the signature of a complementary
signal, not a better similarity function -- the same relationship BM25 has to a
dense retriever. Only in fusion does it come out ahead on both
(+1.5 top-1, +1.5 MRR at the configured weight of 0.25).

The honest caveat: +1.5 top-1 points on 137 pairs is about two queries. It is not
significant and is not presented as such. The test-split experiment with a paired
bootstrap over 300 queries is what decides whether the effect survives, and
:mod:`scripts.train_kernel` fails loudly (exit 2) if fusion ever stops helping.

Interference across query sub-views
-----------------------------------
:meth:`interference_score` superposes several query views *before* measurement,
so cross terms :math:`2\,\mathrm{Re}(\alpha_j\alpha_k z_j \bar z_k)` survive and
one view can cancel another. :meth:`additive_score` measures each view first and
then sums, which is what classical score fusion does and has no cross terms.
Reporting both isolates the interference contribution from the mere effect of
query expansion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import KernelConfig


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau-b between two score vectors, by pair counting."""
    n = len(a)
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    sa = np.sign((a[:, None] - a[None, :])[iu])
    sb = np.sign((b[:, None] - b[None, :])[iu])
    conc = float(np.sum(sa * sb > 0))
    disc = float(np.sum(sa * sb < 0))
    tie_a = float(np.sum((sa == 0) & (sb != 0)))
    tie_b = float(np.sum((sb == 0) & (sa != 0)))
    denom = np.sqrt((conc + disc + tie_a) * (conc + disc + tie_b))
    return float((conc - disc) / denom) if denom else 1.0


# --------------------------------------------------------------- hard negatives
def mine_hard_negatives(query_vecs: np.ndarray, doc_vecs: np.ndarray,
                        positive_rows: np.ndarray, pool_size: int = 64,
                        skip_top: int = 1, chunk: int = 256) -> np.ndarray:
    """Return ``(n_queries, pool_size)`` indices of high-cosine non-positives.

    ``skip_top`` drops the very top hits, which in SciFact are often unjudged
    near-duplicates of the gold abstract; treating those as negatives teaches the
    kernel to push away documents that are in fact relevant.
    """
    n_q = len(query_vecs)
    out = np.zeros((n_q, pool_size), dtype=np.int64)
    want = pool_size + skip_top + 2
    for start in range(0, n_q, chunk):
        stop = min(start + chunk, n_q)
        sims = query_vecs[start:stop] @ doc_vecs.T
        for r in range(stop - start):
            row = sims[r]
            row[positive_rows[start + r]] = -np.inf
            top = np.argpartition(-row, want)[:want]
            top = top[np.argsort(-row[top])]
            out[start + r] = top[skip_top : skip_top + pool_size]
    return out


# ------------------------------------------------------------------ base kernel
class _KernelBase:
    """Shared plumbing: Adam, InfoNCE bookkeeping, save/load, diagnostics."""

    kind = "base"

    def __init__(self, dim: int, cfg: KernelConfig | None = None):
        self.cfg = cfg or KernelConfig()
        self.dim = dim
        self.scale = 1.0
        self.trained = False
        self.history: list[dict] = []

    # -- to be provided by subclasses -------------------------------------
    def params(self) -> list[np.ndarray]:
        raise NotImplementedError

    def set_params(self, values: list[np.ndarray]) -> None:
        raise NotImplementedError

    def _forward(self, Q: np.ndarray, C: np.ndarray):
        raise NotImplementedError

    def _backward(self, cache, dK: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError

    def score(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- shared -----------------------------------------------------------
    def _infonce(self, K: np.ndarray) -> tuple[float, np.ndarray, float]:
        logits = K * (self.scale / self.cfg.temperature)
        logits = logits - logits.max(axis=1, keepdims=True)
        expl = np.exp(logits)
        probs = expl / expl.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(probs[:, 0], 1e-12, None)).mean())
        acc = float((probs.argmax(axis=1) == 0).mean())
        g = probs.copy()
        g[:, 0] -= 1.0
        g *= self.scale / (self.cfg.temperature * K.shape[0])
        return loss, g, acc

    def _calibrate_scale(self, Q: np.ndarray, C: np.ndarray) -> None:
        """Set a fixed gain so the initial logits are neither flat nor saturated.

        The block kernel's raw magnitude depends on the number of blocks, so
        without this the temperature would have to be retuned every time
        ``n_phase_groups`` changes.
        """
        K, _ = self._forward(Q, C)
        sd = float(np.std(K))
        self.scale = 1.0 / sd if sd > 1e-12 else 1.0

    def fit(self, query_vecs: np.ndarray, positive_vecs: np.ndarray,
            negative_vecs: np.ndarray | None = None,
            corpus_vecs: np.ndarray | None = None,
            hard_negatives: np.ndarray | None = None,
            val_fraction: float = 0.15, verbose: bool = True) -> dict:
        """Fit with InfoNCE, early-stopping on a held-out slice of train pairs.

        ``hard_negatives`` is ``(n_pairs, pool)`` document rows into
        ``corpus_vecs``; when absent, negatives are drawn uniformly, which for
        this kernel family is close to no supervision at all.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n = len(query_vecs)
        if n < 8:
            raise ValueError("need at least 8 training pairs")

        perm = rng.permutation(n)
        n_val = max(4, int(n * val_fraction))
        val_rows, train_rows = perm[:n_val], perm[n_val:]

        def batch_candidates(rows: np.ndarray) -> np.ndarray:
            pos = positive_vecs[rows][:, None, :]
            if hard_negatives is not None and corpus_vecs is not None:
                pool = hard_negatives[rows]
                pick = rng.integers(0, pool.shape[1],
                                    size=(len(rows), cfg.n_negatives))
                neg = corpus_vecs[np.take_along_axis(pool, pick, axis=1)]
            else:
                source = corpus_vecs if corpus_vecs is not None else negative_vecs
                neg = source[rng.integers(0, len(source),
                                          size=(len(rows), cfg.n_negatives))]
            return np.concatenate([pos, neg], axis=1)

        self._calibrate_scale(query_vecs[train_rows[:64]],
                             batch_candidates(train_rows[:64]))

        params = self.params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        best = {"val_loss": np.inf, "epoch": -1,
                "params": [p.copy() for p in params]}

        for epoch in range(cfg.epochs):
            order = rng.permutation(train_rows)
            losses, accs = [], []
            for start in range(0, len(order), cfg.batch_size):
                rows = order[start : start + cfg.batch_size]
                if len(rows) < 2:
                    continue
                Q = query_vecs[rows]
                C = batch_candidates(rows)
                K, cache = self._forward(Q, C)
                loss, dK, acc = self._infonce(K)
                grads = self._backward(cache, dK)

                step += 1
                params = self.params()
                new = []
                for i, (p, gr) in enumerate(zip(params, grads)):
                    gr = gr + cfg.weight_decay * p
                    m[i] = b1 * m[i] + (1 - b1) * gr
                    v[i] = b2 * v[i] + (1 - b2) * gr**2
                    mhat = m[i] / (1 - b1**step)
                    vhat = v[i] / (1 - b2**step)
                    new.append(p - cfg.lr * mhat / (np.sqrt(vhat) + eps))
                self.set_params(new)
                losses.append(loss)
                accs.append(acc)

            Qv = query_vecs[val_rows]
            Cv = batch_candidates(val_rows)
            Kv, _ = self._forward(Qv, Cv)
            val_loss, _, val_acc = self._infonce(Kv)
            rec = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                   "top1_acc": float(np.mean(accs)),
                   "val_loss": val_loss, "val_top1_acc": val_acc}
            rec.update(self.param_summary())
            self.history.append(rec)
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "epoch": epoch + 1,
                        "params": [p.copy() for p in self.params()]}
            if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
                print(f"    epoch {rec['epoch']:>3}  train={rec['loss']:.4f} "
                      f"val={val_loss:.4f}  top1={rec['top1_acc']:.3f} "
                      f"val_top1={val_acc:.3f}  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in
                                self.param_summary().items()))

        self.set_params(best["params"])  # early stopping
        self.trained = True
        if verbose:
            print(f"    restored epoch {best['epoch']} "
                  f"(val_loss={best['val_loss']:.4f})")
        return {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                "final": self.history[-1] if self.history else {}}

    def param_summary(self) -> dict:
        return {}

    # -- diagnostics ------------------------------------------------------
    def rank_divergence(self, query_vecs: np.ndarray, doc_vecs: np.ndarray,
                        n_docs: int = 200, seed: int = 0) -> dict:
        """How far this kernel's ranking departs from plain cosine.

        Kendall tau at or near 1.0 means the kernel reorders nothing, and any
        downstream metric change would be noise rather than signal.
        """
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(doc_vecs), size=min(n_docs, len(doc_vecs)),
                            replace=False)
        docs = doc_vecs[sample]
        taus, overlaps = [], []
        for q in query_vecs:
            cos = docs @ q
            ker = self.score(q, docs)
            taus.append(kendall_tau(cos, ker))
            overlaps.append(len(set(np.argsort(-cos)[:10]) &
                                set(np.argsort(-ker)[:10])) / 10.0)
        return {"kendall_tau_mean": float(np.mean(taus)),
                "kendall_tau_min": float(np.min(taus)),
                "kendall_tau_max": float(np.max(taus)),
                "top10_overlap_mean": float(np.mean(overlaps)),
                "n_queries": int(len(query_vecs)), "n_docs": int(len(docs))}

    # -- io ---------------------------------------------------------------
    def save(self, path: Path) -> None:
        blob = {f"p{i}": p for i, p in enumerate(self.params())}
        blob.update(kind=np.array(self.kind), dim=self.dim, scale=self.scale,
                    trained=self.trained)
        np.savez(path, **blob)

    @classmethod
    def load(cls, path: Path, cfg: KernelConfig | None = None):
        blob = np.load(path, allow_pickle=False)
        kind = str(blob["kind"])
        target = {"global-fidelity": GlobalFidelityKernel,
                  "block-fidelity": BlockFidelityKernel}[kind]
        obj = target(int(blob["dim"]), cfg)
        obj.set_params([blob[f"p{i}"] for i in range(len(obj.params()))])
        obj.scale = float(blob["scale"])
        obj.trained = bool(blob["trained"])
        return obj


# ------------------------------------------------------- 1. the negative result
class GlobalFidelityKernel(_KernelBase):
    r"""One global overlap: :math:`|\sum_i q_i d_i e^{i\theta_i}|^2`.

    Retained to demonstrate, rather than assert, that this construction is
    rank-equivalent to cosine at the optimum of a contrastive objective.
    """

    kind = "global-fidelity"

    def __init__(self, dim: int, cfg: KernelConfig | None = None):
        super().__init__(dim, cfg)
        rng = np.random.default_rng(self.cfg.seed)
        self.theta = rng.normal(0.0, 0.05, size=dim)

    def params(self):
        return [self.theta]

    def set_params(self, values):
        self.theta = values[0]

    def param_summary(self):
        return {"abs_theta": float(np.abs(self.theta).mean())}

    def _forward(self, Q, C):
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        U = Q[:, None, :] * C
        A, B = U @ cos_t, U @ sin_t
        return A**2 + B**2, (U, A, B, cos_t, sin_t)

    def _backward(self, cache, dK):
        U, A, B, cos_t, sin_t = cache
        T1 = np.einsum("bm,bmi->i", dK * B, U)
        T2 = np.einsum("bm,bmi->i", dK * A, U)
        return [2.0 * (cos_t * T1 - sin_t * T2)]

    def score(self, query_vec, doc_vecs):
        z = query_vec.astype(np.complex128) * np.exp(1j * self.theta)
        return np.abs(doc_vecs.astype(np.complex128) @ z) ** 2


# ------------------------------------------------------------- 2. the repair
class BlockFidelityKernel(_KernelBase):
    r""":math:`\sum_g w_g |\sum_{i \in g} q_i d_i e^{i\theta_i}|^2`.

    Per-dimension phases (a block-constant phase would factor out and vanish)
    and per-block non-negative weights via softplus.
    """

    kind = "block-fidelity"

    def __init__(self, dim: int, cfg: KernelConfig | None = None):
        super().__init__(dim, cfg)
        g = max(1, min(self.cfg.n_phase_groups, dim))
        while dim % g:  # blocks must tile the embedding exactly
            g -= 1
        self.n_blocks = g
        self.block_size = dim // g
        rng = np.random.default_rng(self.cfg.seed)
        # Phases spread over a full period so that within-block interference is
        # non-trivial from the very first step.
        self.theta = rng.uniform(-np.pi, np.pi, size=dim)
        self.v = np.full(self.n_blocks, 0.5413248546129181)  # softplus(v) == 1

    # -- params -----------------------------------------------------------
    def params(self):
        return [self.theta, self.v]

    def set_params(self, values):
        self.theta, self.v = values[0], values[1]

    @property
    def w(self) -> np.ndarray:
        return softplus(self.v)

    def param_summary(self):
        return {"abs_theta": float(np.abs(self.theta).mean()),
                "w_mean": float(self.w.mean()),
                "w_std": float(self.w.std())}

    # -- forward / backward ----------------------------------------------
    def _blocks(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(*x.shape[:-1], self.n_blocks, self.block_size)

    def _forward(self, Q, C):
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        U = Q[:, None, :] * C                              # (B, M, D)
        A = self._blocks(U * cos_t).sum(-1)                 # (B, M, G)
        B = self._blocks(U * sin_t).sum(-1)
        Kg = A**2 + B**2
        w = self.w
        return Kg @ w, (U, A, B, Kg, cos_t, sin_t, w)

    def _backward(self, cache, dK):
        U, A, B, Kg, cos_t, sin_t, w = cache
        # dL/dw_g, then chain through softplus
        grad_w = np.einsum("bm,bmg->g", dK, Kg)
        grad_v = grad_w * sigmoid(self.v)
        # dL/dtheta_i, i in block g
        GA = np.repeat(dK[..., None] * A * w, self.block_size, axis=-1)
        GB = np.repeat(dK[..., None] * B * w, self.block_size, axis=-1)
        grad_theta = 2.0 * np.einsum("bmi,bmi->i", U, GB * cos_t - GA * sin_t)
        return [grad_theta, grad_v]

    # -- inference --------------------------------------------------------
    def score(self, query_vec, doc_vecs):
        U = doc_vecs * query_vec
        A = self._blocks(U * np.cos(self.theta)).sum(-1)
        B = self._blocks(U * np.sin(self.theta)).sum(-1)
        return (A**2 + B**2) @ self.w

    def cos2_score(self, query_vec, doc_vecs):
        """Cosine squared, for the rank-equivalence demonstration."""
        return (doc_vecs @ query_vec) ** 2

    # -- interference -----------------------------------------------------
    def _subquery_weights(self, n: int) -> np.ndarray:
        w = np.array([self.cfg.subquery_weight_decay**j for j in range(n)])
        return w / np.linalg.norm(w)

    def _block_amplitudes(self, query_views: np.ndarray,
                          doc_vecs: np.ndarray) -> np.ndarray:
        """Complex per-block amplitude, shape ``(n_views, n_docs, n_blocks)``."""
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        amps = []
        for view in query_views:
            U = doc_vecs * view
            A = self._blocks(U * cos_t).sum(-1)
            B = self._blocks(U * sin_t).sum(-1)
            amps.append(A + 1j * B)
        return np.stack(amps)

    def interference_score(self, query_views, doc_vecs):
        """Superpose views, then measure. Cross terms survive."""
        alpha = self._subquery_weights(len(query_views))
        amps = self._block_amplitudes(query_views, doc_vecs)
        combined = np.tensordot(alpha, amps, axes=(0, 0))     # (N, G)
        return (np.abs(combined) ** 2) @ self.w

    def additive_score(self, query_views, doc_vecs):
        """Measure each view, then sum. The classical comparator, no cross terms."""
        alpha = self._subquery_weights(len(query_views))
        amps = self._block_amplitudes(query_views, doc_vecs)
        per_view = (np.abs(amps) ** 2) @ self.w                # (J, N)
        return (alpha**2) @ per_view


# Default export used by the pipeline.
PhaseKernel = BlockFidelityKernel
