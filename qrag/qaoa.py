r"""QAOA reranking: top-k context selection as combinatorial optimisation.

Why this is the strongest of the three quantum components
---------------------------------------------------------
A classical retriever sorts by relevance and takes the top k. That greedy sort
is *provably* not optimising anything about the selected set as a whole -- it
cannot trade a slightly less relevant document for one that adds information the
others do not already carry. Written down properly, choosing a context set is a
constrained quadratic binary problem:

.. math::
    \min_{x \in \{0,1\}^n} \; -\sum_i r_i x_i
        \;+\; \lambda \sum_{i<j} s_{ij} x_i x_j
        \;+\; \mu \Big(\sum_i x_i - k\Big)^2

where :math:`r_i` is fused relevance, :math:`s_{ij}` is inter-document
similarity, and the last term is a soft cardinality penalty. This is a QUBO, so
it maps directly onto QAOA -- and unlike the Grover component, there is a real
combinatorial objective here that the classical baseline genuinely fails to
optimise. Because :math:`n \le 16`, the exact optimum is also computable by
enumeration, which lets us report a true approximation ratio rather than a
relative improvement against another heuristic.

The security hypothesis
-----------------------
The :math:`\lambda \sum s_{ij} x_i x_j` term penalises selecting mutually
similar documents. Injected adversarial passages that target one query tend to
be mutually similar (they are all optimised toward the same embedding
neighbourhood), so the redundancy penalty should suppress *clusters* of them
even when each individual passage scores well. If that holds, QAOA reranking is
a retrieval-level defence that falls out of the optimisation formulation rather
than a bolted-on filter. :func:`selection_redundancy` measures the quantity the
hypothesis is about.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .qsim import Statevector, all_bitstrings
from .security import check_qubit_budget


@dataclass
class QUBO:
    """Cost model over ``n`` candidate documents."""

    relevance: np.ndarray            # (n,) fused relevance, min-max normalised
    similarity: np.ndarray           # (n, n) symmetric, zero diagonal
    k: int
    redundancy_lambda: float = 0.55
    cardinality_mu: float = 1.5

    @property
    def n(self) -> int:
        return len(self.relevance)

    def linear(self) -> np.ndarray:
        mu, k = self.cardinality_mu, self.k
        return -self.relevance + mu * (1.0 - 2.0 * k)

    def quadratic(self) -> np.ndarray:
        """Symmetric ``W`` with ``x^T W x == sum_{i<j} Q_ij x_i x_j``."""
        n = self.n
        q = self.redundancy_lambda * self.similarity + 2.0 * self.cardinality_mu
        np.fill_diagonal(q, 0.0)
        w = q / 2.0
        return (w + w.T) / 2.0

    def offset(self) -> float:
        return self.cardinality_mu * self.k**2

    def energies(self) -> np.ndarray:
        """Cost of every one of the ``2^n`` assignments, as a flat array.

        This is the diagonal of the cost Hamiltonian, which is all QAOA needs --
        and at n <= 16 it also gives us the exact optimum for free.
        """
        x = all_bitstrings(self.n).astype(np.float64)
        lin = x @ self.linear()
        quad = np.einsum("ni,ij,nj->n", x, self.quadratic(), x)
        return lin + quad + self.offset()

    def objective(self, selection: np.ndarray) -> float:
        """Cost of one selection, given as a list of selected indices.

        Indices only. An earlier docstring claimed a 0/1 vector was also accepted;
        it is not, and passing one is silently wrong -- ``x[[0,1,1,0]] = 1`` sets
        positions 0 and 1 rather than the positions where the vector is 1. Callers
        that hold a bit vector must convert with ``np.flatnonzero`` first.
        """
        x = np.zeros(self.n)
        x[np.asarray(selection, dtype=int)] = 1.0
        return float(x @ self.linear() + x @ self.quadratic() @ x + self.offset())


@dataclass
class RerankResult:
    method: str
    selected: list[int]
    objective: float
    wall_clock_s: float
    # QAOA-specific diagnostics; None for classical methods
    n_qubits: int | None = None
    layers: int | None = None
    optimiser_iters: int | None = None
    feasible_probability: float | None = None
    approximation_ratio: float | None = None
    extras: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = f"{self.method:<16} obj={self.objective:+.4f}"
        if self.approximation_ratio is not None:
            head += f" approx={self.approximation_ratio:.4f}"
        head += f" sel={self.selected} {self.wall_clock_s * 1e3:.1f} ms"
        return head


# --------------------------------------------------------------- classical refs
def rerank_topk(qubo: QUBO) -> RerankResult:
    """Sort by relevance, take k. This is what the classical baseline does."""
    t0 = time.perf_counter()
    sel = sorted(np.argsort(-qubo.relevance)[: qubo.k].tolist())
    return RerankResult("topk-sort", sel, qubo.objective(sel),
                        time.perf_counter() - t0)


def rerank_greedy_mmr(qubo: QUBO) -> RerankResult:
    """Maximal Marginal Relevance: the standard greedy diversity heuristic."""
    t0 = time.perf_counter()
    chosen: list[int] = []
    remaining = set(range(qubo.n))
    while len(chosen) < qubo.k and remaining:
        best, best_gain = None, -np.inf
        for i in remaining:
            penalty = (max(qubo.similarity[i, j] for j in chosen) if chosen else 0.0)
            gain = qubo.relevance[i] - qubo.redundancy_lambda * penalty
            if gain > best_gain:
                best, best_gain = i, gain
        chosen.append(int(best))
        remaining.discard(best)
    sel = sorted(chosen)
    return RerankResult("greedy-mmr", sel, qubo.objective(sel),
                        time.perf_counter() - t0)


def feasible_bounds(energies: np.ndarray, n: int, k: int) -> tuple[float, float]:
    """Best and worst objective over the exactly-``k`` selections.

    Both are needed because the ratio has to be affine-invariant -- see
    :func:`solution_quality`. Reads them off the already-computed cost diagonal,
    so it costs one mask rather than a second enumeration.
    """
    bits = all_bitstrings(n)
    feasible = energies[bits.sum(axis=1) == k]
    return float(feasible.min()), float(feasible.max())


def solution_quality(achieved: float, optimal: float, worst: float) -> float | None:
    r"""Normalised solution quality in :math:`[0, 1]`, where 1 is optimal.

    .. math:: q = \frac{f_\text{worst} - f}{f_\text{worst} - f_\text{opt}}

    The obvious definition, ``achieved / optimal``, is what this replaces, and it
    was wrong. This QUBO's objective reduces to
    :math:`-\sum_i r_i + \lambda \sum_{i<j} S_{ij}` on feasible selections, which is
    negative when relevance dominates but **positive** when a query's candidates are
    weakly relevant and mutually similar. In that regime a *worse* selection divided
    by a positive optimum gives a number above 1, so a ratio of 1.0058 was being
    reported and read as "better than the exact optimum" -- which is impossible, as
    both search the same exactly-k set. It meant QAOA was 0.58% worse.

    The affine-invariant form is immune to that: any offset or positive rescaling of
    the objective leaves it unchanged, and it is monotone in solution goodness
    whatever the sign. Degenerate case: if every feasible selection has the same
    cost there is nothing to choose between them, and this returns None rather than
    dividing by zero and claiming a perfect score.
    """
    spread = worst - optimal
    if spread <= 1e-12:
        return None
    return float((worst - achieved) / spread)


def rerank_exact(qubo: QUBO) -> RerankResult:
    """Exact optimum by enumerating all C(n, k) feasible selections."""
    # C(n, k) <= 2^n, so the enumeration ceiling bounds this too. The guard is
    # explicit rather than inherited because this function does not go through
    # all_bitstrings and would otherwise be an unbounded loop over candidates.
    check_qubit_budget(qubo.n, exact=True)
    t0 = time.perf_counter()
    best, best_obj = None, np.inf
    for combo in itertools.combinations(range(qubo.n), qubo.k):
        obj = qubo.objective(np.array(combo))
        if obj < best_obj:
            best, best_obj = combo, obj
    return RerankResult("exact-bruteforce", sorted(best), float(best_obj),
                        time.perf_counter() - t0)


# ------------------------------------------------------------------------ QAOA
def qaoa_expectation(energies: np.ndarray, n_qubits: int,
                     gammas: np.ndarray, betas: np.ndarray) -> tuple[float, Statevector]:
    r"""Prepare the QAOA ansatz and return :math:`\langle H_C \rangle`."""
    state = Statevector.uniform(n_qubits)
    for gamma, beta in zip(gammas, betas):
        state.apply_diagonal_phase(energies, gamma)
        state.apply_rx_all(2.0 * beta)
    return state.expectation_diagonal(energies), state


def rerank_qaoa(qubo: QUBO, layers: int = 2, optimiser_iters: int = 120,
                seed: int = 20260720, backend: str = "numpy",
                exact_objective: float | None = None) -> RerankResult:
    """Solve the selection QUBO with QAOA and decode the best feasible state."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    n = qubo.n
    energies = qubo.energies()
    # Rescale so the phase separator does not wrap for large penalty weights;
    # this is a reparameterisation of gamma, not a change of objective.
    scale = float(np.max(np.abs(energies))) or 1.0
    scaled = energies / scale

    rng = np.random.default_rng(seed)
    calls = {"n": 0}

    def negated(params: np.ndarray) -> float:
        calls["n"] += 1
        gammas, betas = params[:layers], params[layers:]
        value, _ = qaoa_expectation(scaled, n, gammas, betas)
        return value  # minimising the cost Hamiltonian directly

    best_params, best_value = None, np.inf
    for _restart in range(3):  # QAOA landscapes are non-convex; restart a few times
        x0 = np.concatenate([rng.uniform(0, np.pi, layers),
                             rng.uniform(0, np.pi / 2, layers)])
        res = minimize(negated, x0, method="COBYLA",
                       options={"maxiter": optimiser_iters})
        if res.fun < best_value:
            best_params, best_value = res.x, float(res.fun)

    gammas, betas = best_params[:layers], best_params[layers:]
    if backend == "aer":
        probs = _qaoa_probs_aer(qubo, scaled, gammas, betas)
    else:
        _, state = qaoa_expectation(scaled, n, gammas, betas)
        probs = state.probabilities()

    # Decode: restrict to selections of the right cardinality, then take the
    # most probable. The cardinality term is a soft penalty, so infeasible
    # states retain some amplitude and must be filtered at readout.
    bits = all_bitstrings(n)
    cardinality = bits.sum(axis=1)
    feasible = np.flatnonzero(cardinality == qubo.k)
    feasible_mass = float(probs[feasible].sum())
    best_idx = int(feasible[np.argmax(probs[feasible])])
    selected = sorted(np.flatnonzero(bits[best_idx]).tolist())
    objective = qubo.objective(selected)

    # Bounds over the same feasible set QAOA just decoded from, so the quality
    # figure is computed against a like-for-like optimum rather than a value
    # threaded in from elsewhere. When the caller supplies exact_objective anyway,
    # the two must agree -- a mismatch means the exact solver and the decoder are
    # not optimising the same problem, which is worth failing loudly for.
    optimal, worst = feasible_bounds(energies, n, qubo.k)
    if exact_objective is not None and abs(exact_objective - optimal) > 1e-6:
        raise AssertionError(
            f"exact optimum disagrees with the feasible-set minimum: "
            f"{exact_objective:.10f} vs {optimal:.10f}")
    quality = solution_quality(objective, optimal, worst)

    return RerankResult(
        method=f"qaoa-p{layers}",
        selected=selected,
        objective=objective,
        wall_clock_s=time.perf_counter() - t0,
        n_qubits=n,
        layers=layers,
        optimiser_iters=calls["n"],
        feasible_probability=feasible_mass,
        approximation_ratio=quality,
        extras={
            "gammas": gammas.tolist(),
            "betas": betas.tolist(),
            "energy_scale": scale,
            "expectation_scaled": best_value,
            "backend": backend,
            # Reported so the quality figure can be audited, and so "optimal" is
            # visible as a value rather than only as a ratio of 1.0.
            "optimal_objective": optimal,
            "worst_feasible_objective": worst,
            "objective_gap": float(objective - optimal),
            "is_optimal": bool(objective <= optimal + 1e-9),
        },
    )


def _qaoa_probs_aer(qubo: QUBO, scaled: np.ndarray, gammas, betas) -> np.ndarray:
    """Gate-level QAOA on Aer, used to validate the numpy fast path.

    The diagonal cost operator is decomposed into RZ (linear terms) and RZZ
    (quadratic terms), which is the standard Ising translation of a QUBO.
    """
    # Aer allocates its own 2**n statevector outside our simulator's ceiling.
    check_qubit_budget(qubo.n, exact=True)
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    n = qubo.n
    scale = float(np.max(np.abs(qubo.energies()))) or 1.0
    # x = (1 - z)/2 substitution to move from binary to Ising variables.
    h = qubo.linear() / scale
    w = qubo.quadratic() / scale
    h_z = -0.5 * h - 0.5 * w.sum(axis=1)
    j_z = 0.5 * w

    qc = QuantumCircuit(n)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n):
            if abs(h_z[i]) > 1e-12:
                qc.rz(2.0 * gamma * h_z[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(j_z[i, j]) > 1e-12:
                    qc.rzz(2.0 * gamma * j_z[i, j], i, j)
        qc.rx(2.0 * beta, range(n))
    qc.save_statevector()

    sim = AerSimulator(method="statevector")
    result = sim.run(transpile(qc, sim)).result()
    return np.abs(np.asarray(result.get_statevector())) ** 2


# -------------------------------------------------------------------- analysis
def selection_redundancy(qubo: QUBO, selected: list[int]) -> float:
    """Mean pairwise similarity within a selected set.

    The quantity behind the security hypothesis: lower means the context window
    holds less duplicated material, and therefore fewer slots consumed by a
    cluster of near-identical injected passages.
    """
    if len(selected) < 2:
        return 0.0
    pairs = [qubo.similarity[i, j] for i, j in itertools.combinations(selected, 2)]
    return float(np.mean(pairs))


def compare_rerankers(qubo: QUBO, layers: int = 2, optimiser_iters: int = 120,
                      seed: int = 20260720, include_aer: bool = False) -> list[RerankResult]:
    """Run every reranker on the same QUBO for a like-for-like comparison."""
    exact = rerank_exact(qubo)
    optimal, worst = feasible_bounds(qubo.energies(), qubo.n, qubo.k)
    results = [rerank_topk(qubo), rerank_greedy_mmr(qubo), exact,
               rerank_qaoa(qubo, layers, optimiser_iters, seed, "numpy",
                           exact.objective)]
    if include_aer:
        results.append(rerank_qaoa(qubo, layers, optimiser_iters, seed, "aer",
                                   exact.objective))
    for r in results:
        # Same affine-invariant definition for every method, so topk-sort,
        # greedy-MMR and QAOA are scored on one scale. The old division by
        # exact.objective inverted whenever that objective was positive.
        if r.approximation_ratio is None:
            r.approximation_ratio = solution_quality(r.objective, optimal, worst)
        r.extras["redundancy"] = selection_redundancy(qubo, r.selected)
    return results


def validate_against_aer(n: int = 8, k: int = 3, layers: int = 2,
                         seed: int = 0, tol: float = 1e-8) -> dict:
    """Confirm the numpy QAOA ansatz matches the gate-level Aer circuit."""
    rng = np.random.default_rng(seed)
    rel = rng.random(n)
    sim = rng.random((n, n))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 0.0)
    qubo = QUBO(rel, sim, k)

    energies = qubo.energies()
    scale = float(np.max(np.abs(energies))) or 1.0
    gammas = rng.uniform(0, np.pi, layers)
    betas = rng.uniform(0, np.pi / 2, layers)

    _, state = qaoa_expectation(energies / scale, n, gammas, betas)
    p_numpy = state.probabilities()
    p_aer = _qaoa_probs_aer(qubo, energies / scale, gammas, betas)

    diff = float(np.max(np.abs(p_numpy - p_aer)))
    return {"n_qubits": n, "layers": layers, "max_abs_prob_diff": diff,
            "agrees": diff < tol}


if __name__ == "__main__":
    print("QAOA numpy vs Aer:", validate_against_aer())
    rng = np.random.default_rng(1)
    rel = rng.random(12)
    s = rng.random((12, 12)); s = (s + s.T) / 2; np.fill_diagonal(s, 0)
    q = QUBO(rel, s, k=5)
    for r in compare_rerankers(q):
        print(" ", r.summary(), "redundancy=%.4f" % r.extras["redundancy"])
