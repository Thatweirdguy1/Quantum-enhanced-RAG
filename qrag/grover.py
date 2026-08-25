r"""Grover-style amplitude amplification over a retrieval shortlist.

Role in the pipeline
--------------------
Grover's algorithm needs an oracle that recognises a marked item. In retrieval
that oracle is "is this document's relevance above threshold :math:`\tau`",
which presupposes a score -- so amplitude amplification cannot search the raw
corpus. What it *can* do is narrow an already-scored shortlist, and that is how
it is used here: classical fusion scores a shortlist of :math:`N = 2^n`
candidates, a quantile threshold marks the promising ones, and amplification
concentrates amplitude on the marked set in
:math:`\lfloor \frac{\pi}{4}\sqrt{N/M} \rfloor` oracle queries instead of the
:math:`(N{+}1)/(M{+}1)` a classical random probe needs in expectation.

HOW TO REPORT THIS HONESTLY
---------------------------
Statevector simulation of an n-qubit Grover circuit costs :math:`O(2^n)`
classically, so the *wall-clock* here is always worse than the classical scan it
is nominally accelerating. The quadratic speedup is a statement about oracle
query complexity on real quantum hardware, not about the runtime of a simulator.
:class:`GroverResult` therefore carries the two quantities in separate fields --
``oracle_queries`` / ``classical_expected_queries`` (where the
:math:`O(\sqrt{N})` claim legitimately lives, as a projected complexity result)
and ``wall_clock_s`` / ``simulation_overhead`` (a measured cost of simulation,
with no speedup claimed). Any table in the report must keep them in separate
columns and say which is which.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .qsim import Statevector
from .security import check_qubit_budget


@dataclass
class GroverResult:
    n_qubits: int
    n_candidates: int
    n_marked: int
    iterations: int
    # --- query-complexity axis: the legitimate home of the sqrt(N) claim
    oracle_queries: int
    classical_expected_queries: float
    query_reduction_factor: float
    # --- measured-runtime axis: simulation cost, no speedup claimed
    wall_clock_s: float
    classical_scan_s: float
    simulation_overhead: float
    # --- retrieval outcome
    success_probability: float
    ranked_indices: list[int] = field(default_factory=list)
    backend: str = "numpy"

    def summary(self) -> str:
        return (
            f"Grover[{self.backend}] n={self.n_qubits}q N={self.n_candidates} "
            f"M={self.n_marked} iters={self.iterations} "
            f"P(marked)={self.success_probability:.4f} | "
            f"queries {self.oracle_queries} vs classical "
            f"{self.classical_expected_queries:.1f} "
            f"({self.query_reduction_factor:.2f}x fewer) | "
            f"sim wall-clock {self.wall_clock_s * 1e3:.2f} ms "
            f"({self.simulation_overhead:.1f}x the classical scan)"
        )


def optimal_iterations(n_candidates: int, n_marked: int) -> int:
    if n_marked <= 0 or n_marked >= n_candidates:
        return 0
    theta = math.asin(math.sqrt(n_marked / n_candidates))
    return max(1, int(round((math.pi / 2 - theta) / (2 * theta))))


def _pad_to_power_of_two(scores: np.ndarray) -> tuple[np.ndarray, int, int]:
    n_real = len(scores)
    n_qubits = max(1, math.ceil(math.log2(max(n_real, 2))))
    size = 2**n_qubits
    padded = np.full(size, -np.inf, dtype=np.float64)
    padded[:n_real] = scores
    return padded, n_qubits, n_real


class GroverShortlister:
    """Amplitude amplification over a threshold oracle on shortlist scores."""

    def __init__(self, threshold_quantile: float = 0.80, backend: str = "numpy"):
        self.threshold_quantile = threshold_quantile
        self.backend = backend

    def _marked_set(self, padded: np.ndarray, n_real: int) -> tuple[np.ndarray, float]:
        real = padded[:n_real]
        tau = float(np.quantile(real, self.threshold_quantile))
        marked = np.flatnonzero(padded >= tau)
        if len(marked) == 0:  # degenerate: mark the single best candidate
            marked = np.array([int(np.argmax(padded))])
            tau = float(padded[marked[0]])
        return marked, tau

    def run(self, scores: np.ndarray, iterations: int | None = None) -> GroverResult:
        padded, n_qubits, n_real = _pad_to_power_of_two(np.asarray(scores, float))
        size = 2**n_qubits
        marked, _tau = self._marked_set(padded, n_real)
        n_marked = len(marked)
        iters = optimal_iterations(size, n_marked) if iterations is None else iterations

        # Measured classical reference: a full scan of the same shortlist.
        t0 = time.perf_counter()
        _ = int(np.argmax(padded))
        classical_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.backend == "aer":
            probs = self._run_aer(n_qubits, marked, iters)
        else:
            state = Statevector.uniform(n_qubits)
            for _ in range(iters):
                state.flip_phase(marked)
                state.grover_diffuser()
            probs = state.probabilities()
        wall = time.perf_counter() - t0

        success = float(probs[marked].sum())
        order = np.argsort(-probs[:n_real])
        classical_expected = (size + 1) / (n_marked + 1)

        return GroverResult(
            n_qubits=n_qubits,
            n_candidates=size,
            n_marked=n_marked,
            iterations=iters,
            oracle_queries=iters,
            classical_expected_queries=classical_expected,
            query_reduction_factor=classical_expected / max(iters, 1),
            wall_clock_s=wall,
            classical_scan_s=classical_s,
            simulation_overhead=wall / classical_s if classical_s > 0 else float("nan"),
            success_probability=success,
            ranked_indices=[int(i) for i in order],
            backend=self.backend,
        )

    # ------------------------------------------------------------------ qiskit
    def _run_aer(self, n_qubits: int, marked: np.ndarray, iters: int) -> np.ndarray:
        """Gate-level construction, executed on the Aer statevector backend."""
        # The numpy path is bounded by Statevector; Aer allocates its own 2**n
        # buffer and would otherwise bypass the ceiling entirely.
        check_qubit_budget(n_qubits)
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        marked_bits = [format(int(m), f"0{n_qubits}b")[::-1] for m in marked]

        for _ in range(iters):
            # Oracle: a multi-controlled Z per marked basis state, conjugated by
            # X gates so the control pattern selects that state.
            for bits in marked_bits:
                zeros = [i for i, b in enumerate(bits) if b == "0"]
                if zeros:
                    qc.x(zeros)
                if n_qubits == 1:
                    qc.z(0)
                else:
                    qc.h(n_qubits - 1)
                    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                    qc.h(n_qubits - 1)
                if zeros:
                    qc.x(zeros)
            # Diffuser
            qc.h(range(n_qubits))
            qc.x(range(n_qubits))
            if n_qubits == 1:
                qc.z(0)
            else:
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)
            qc.x(range(n_qubits))
            qc.h(range(n_qubits))

        qc.save_statevector()
        sim = AerSimulator(method="statevector")
        result = sim.run(transpile(qc, sim)).result()
        return np.abs(np.asarray(result.get_statevector())) ** 2


def validate_against_aer(n_qubits: int = 4, marked=(3, 9), tol: float = 1e-9) -> dict:
    """Confirm the numpy diffuser matches the gate-level Aer circuit.

    The O(2^n) mean-reflection shortcut in :meth:`Statevector.grover_diffuser`
    is algebraically equal to H-MCZ-H; this asserts it numerically so the fast
    path is a verified optimisation rather than an assumption.
    """
    marked = np.asarray(marked)
    iters = optimal_iterations(2**n_qubits, len(marked))

    state = Statevector.uniform(n_qubits)
    for _ in range(iters):
        state.flip_phase(marked)
        state.grover_diffuser()
    p_numpy = state.probabilities()

    shortlister = GroverShortlister(backend="aer")
    p_aer = shortlister._run_aer(n_qubits, marked, iters)

    max_abs_diff = float(np.max(np.abs(p_numpy - p_aer)))
    return {
        "n_qubits": n_qubits,
        "iterations": iters,
        "max_abs_prob_diff": max_abs_diff,
        "agrees": max_abs_diff < tol,
        "success_prob_numpy": float(p_numpy[marked].sum()),
        "success_prob_aer": float(p_aer[marked].sum()),
    }


if __name__ == "__main__":
    print("numpy vs Aer:", validate_against_aer())
    rng = np.random.default_rng(0)
    print(GroverShortlister().run(rng.random(64)).summary())
