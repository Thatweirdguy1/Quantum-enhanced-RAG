r"""A small exact statevector simulator.

Both quantum components in this project need only three primitives: uniform
superposition, a diagonal phase operator, and a transverse-field mixer. All
three have closed forms on the amplitude array, so a dedicated 150-line
simulator is faster than routing through a general circuit framework and makes
the linear algebra visible for the report.

:mod:`qrag.grover` and :mod:`qrag.qaoa` cross-check every result here against
qiskit-aer, so this module is a fast path rather than an unverified shortcut.

Qubit ordering is little-endian (qubit 0 is the least significant bit), matching
Qiskit, so bitstrings can be compared between backends without reversal bugs.
"""

from __future__ import annotations

import numpy as np

from .security import check_qubit_budget

# Kept as a module constant for readability; the enforced ceiling is
# SecurityConfig.max_qubits (env: QRAG_MAX_QUBITS) so that it is configurable in
# one place alongside the other resource limits.
MAX_QUBITS = 22  # 2**22 complex128 = 64 MiB; refuse beyond this


class Statevector:
    def __init__(self, n_qubits: int, data: np.ndarray | None = None):
        # A statevector is 2**n complex numbers, so an unvalidated qubit count is
        # a one-line memory exhaustion. Exact simulation not scaling is the
        # NISQ-era wall this project measures, not a bug -- but it must fail with
        # an explanation rather than by taking the machine down.
        check_qubit_budget(n_qubits)
        self.n = n_qubits
        self.dim = 2**n_qubits
        if data is None:
            data = np.zeros(self.dim, dtype=np.complex128)
            data[0] = 1.0
        self.data = data.astype(np.complex128)

    # ------------------------------------------------------------ constructors
    @classmethod
    def uniform(cls, n_qubits: int) -> Statevector:
        r"""The state :math:`H^{\otimes n}|0\rangle`."""
        # Validate before np.full: the array is an argument, so it would be
        # allocated before __init__ got the chance to refuse it.
        check_qubit_budget(n_qubits)
        dim = 2**n_qubits
        return cls(n_qubits, np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128))

    # ------------------------------------------------------------------- gates
    def apply_1q(self, gate: np.ndarray, qubit: int) -> Statevector:
        """Apply a 2x2 unitary to ``qubit`` without materialising a 2^n matrix."""
        if not 0 <= qubit < self.n:
            raise IndexError(qubit)
        lo = 2**qubit
        hi = 2 ** (self.n - qubit - 1)
        view = self.data.reshape(hi, 2, lo)
        a, b = view[:, 0, :].copy(), view[:, 1, :].copy()
        view[:, 0, :] = gate[0, 0] * a + gate[0, 1] * b
        view[:, 1, :] = gate[1, 0] * a + gate[1, 1] * b
        return self

    def apply_rx_all(self, angle: float) -> Statevector:
        r"""The QAOA mixer :math:`\exp(-i\beta \sum_j X_j)` as a product of RX."""
        c, s = np.cos(angle / 2.0), -1j * np.sin(angle / 2.0)
        gate = np.array([[c, s], [s, c]], dtype=np.complex128)
        for q in range(self.n):
            self.apply_1q(gate, q)
        return self

    def apply_diagonal_phase(self, energies: np.ndarray, gamma: float) -> Statevector:
        r"""Apply :math:`\exp(-i\gamma H_C)` for diagonal :math:`H_C`.

        A cost Hamiltonian that is diagonal in the computational basis is exactly
        a per-amplitude phase, so this is one vectorised multiply regardless of
        how many terms the underlying QUBO has.
        """
        self.data *= np.exp(-1j * gamma * energies)
        return self

    def flip_phase(self, marked: np.ndarray) -> Statevector:
        r"""Grover oracle: :math:`|x\rangle \mapsto -|x\rangle` for marked x."""
        self.data[marked] *= -1.0
        return self

    def grover_diffuser(self) -> Statevector:
        r"""Apply :math:`2|s\rangle\langle s| - I` in O(2^n).

        Algebraically identical to the H-MCZ-H gate sequence, and validated
        against it in :func:`qrag.grover.validate_against_aer`.
        """
        self.data = 2.0 * self.data.mean() - self.data
        return self

    # -------------------------------------------------------------- measurement
    def probabilities(self) -> np.ndarray:
        return np.abs(self.data) ** 2

    def expectation_diagonal(self, energies: np.ndarray) -> float:
        return float(np.dot(self.probabilities(), energies))

    def sample(self, shots: int, seed: int = 0) -> dict[int, int]:
        rng = np.random.default_rng(seed)
        p = self.probabilities()
        p = p / p.sum()
        draws = rng.choice(self.dim, size=shots, p=p)
        vals, counts = np.unique(draws, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def top_states(self, n: int = 5) -> list[tuple[str, float]]:
        p = self.probabilities()
        idx = np.argsort(-p)[:n]
        return [(format(int(i), f"0{self.n}b"), float(p[i])) for i in idx]

    def norm(self) -> float:
        return float(np.linalg.norm(self.data))


def bits_of(index: int, n: int) -> np.ndarray:
    """Little-endian bit array for a basis-state index."""
    return np.array([(index >> b) & 1 for b in range(n)], dtype=np.int8)


def all_bitstrings(n: int) -> np.ndarray:
    """``(2^n, n)`` matrix of every bit assignment, little-endian.

    Guarded by the tighter ``exact`` ceiling: this materialises ``2^n * n`` bytes
    rather than a ``2^n`` amplitude vector, so it hits memory limits sooner than
    the simulator does.
    """
    check_qubit_budget(n, exact=True)
    idx = np.arange(2**n, dtype=np.uint64)
    return ((idx[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(np.int8)
