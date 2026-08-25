"""Experiment configuration.

Every run is fully described by a :class:`Config`. The config is hashed into the
results log so that any number appearing in the report can be traced back to the
exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)


@dataclass
class EmbedConfig:
    # "ollama" uses the local daemon (no torch dependency); "sbert" uses the
    # sentence-transformers backend named in the synopsis; "hashing" is a
    # deterministic offline fallback used only for CI smoke tests.
    backend: str = os.environ.get("QRAG_EMBED_BACKEND", "ollama")
    model: str = os.environ.get("QRAG_EMBED_MODEL", "bge-m3")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    dim: int = 1024
    batch_size: int = 16
    max_chars: int = 2000  # truncate long abstracts before embedding


@dataclass
class KernelConfig:
    """Phase-modulated fidelity kernel.

    ``n_phase_groups`` ties phases in blocks rather than learning one per
    dimension. Blocking acts as a regulariser: SciFact's train split has only
    ~900 judgments, which is thin for 1024 free parameters.
    """

    n_phase_groups: int = 128
    temperature: float = 0.05
    lr: float = 0.05
    epochs: int = 40
    batch_size: int = 32
    n_negatives: int = 15
    weight_decay: float = 1e-4
    seed: int = 20260720  # week 1 start date, used as the global seed
    # Interference scoring: number of query sub-views superposed before
    # measurement. 1 disables interference and falls back to the plain kernel.
    n_subqueries: int = 3
    subquery_weight_decay: float = 0.6


@dataclass
class FusionConfig:
    w_bm25: float = 0.30
    w_cosine: float = 0.45
    w_kernel: float = 0.25
    # Scores are min-max normalised per query before mixing, since BM25 and
    # cosine live on incomparable scales.
    normalise: str = "minmax"


@dataclass
class QuantumConfig:
    backend: str = "numpy"  # "numpy" (qrag.qsim) or "aer" (qiskit-aer)
    # Grover
    grover_shortlist: int = 64  # candidates handed to amplitude amplification
    grover_threshold_quantile: float = 0.80
    # QAOA reranking
    qaoa_candidates: int = 12  # == qubit count, keep <= 16 for statevector sim
    qaoa_k: int = 5  # documents selected into the final context
    qaoa_layers: int = 2  # circuit depth p
    qaoa_redundancy_lambda: float = 0.55
    qaoa_cardinality_mu: float = 1.5
    qaoa_optimiser_iters: int = 120
    seed: int = 20260720


@dataclass
class GenConfig:
    backend: str = "ollama"
    model: str = os.environ.get("QRAG_GEN_MODEL", "qwen3.5")
    # Deliberately a different model family from the generator: using the
    # generator to grade itself introduces well-documented self-preference bias.
    judge_model: str = os.environ.get("QRAG_JUDGE_MODEL", "deepseek-r1:8b")
    temperature: float = 0.0
    max_context_docs: int = 5


@dataclass
class EvalConfig:
    dataset: str = "scifact"
    split: str = "test"
    k_values: tuple = (1, 3, 5, 10, 20)
    primary_k: int = 10
    n_queries: int | None = None  # None = all queries in the split
    bootstrap_samples: int = 2000
    seed: int = 20260720


@dataclass
class Config:
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf8")


DEFAULT = Config()
