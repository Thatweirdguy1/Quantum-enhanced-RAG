"""Load the cached Q-RAG stack used by the demo API."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CACHE_DIR, Config, DEFAULT
from .data import Dataset, load_beir
from .embed import Embedder, build_embedder
from .generate import OllamaGenerator, OllamaJudge
from .kernel import PhaseKernel
from .pipeline import QRAG, build_indexes


@dataclass
class Runtime:
    pipeline: QRAG
    dataset: Dataset
    embedder: Embedder
    generator: OllamaGenerator
    judge: OllamaJudge


def load_runtime(cfg: Config = DEFAULT) -> Runtime:
    dataset = load_beir(cfg.eval.dataset, cfg.eval.split)
    embedder = build_embedder(cfg.embed)
    vectors = embedder.encode_cached(
        [doc.content for doc in dataset.documents], f"{cfg.eval.dataset}-corpus")
    kernel_path = CACHE_DIR / "phase_kernel.npz"
    if not kernel_path.exists():
        raise FileNotFoundError(f"{kernel_path} not found; run scripts.train_kernel")
    kernel = PhaseKernel.load(kernel_path, cfg.kernel)
    if not kernel.trained:
        raise RuntimeError("cached phase kernel is not trained")
    dense, bm25 = build_indexes(dataset, vectors)
    pipeline = QRAG(dataset, dense, bm25, {}, kernel, cfg)
    return Runtime(pipeline, dataset, embedder, OllamaGenerator(cfg.gen),
                   OllamaJudge(cfg.gen))
