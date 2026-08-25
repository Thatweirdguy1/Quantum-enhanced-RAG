"""Pre-compute and cache corpus + query embeddings.

Run this once before any experiment:

    python -m scripts.build_embeddings

Embedding 5,183 SciFact abstracts through a local bge-m3 takes a few minutes;
every downstream experiment then loads from ``cache/`` in under a second.
"""

from __future__ import annotations

import time

from qrag.config import DEFAULT
from qrag.data import load_beir, load_train_pairs
from qrag.embed import build_embedder


def main() -> None:
    cfg = DEFAULT
    t0 = time.time()

    ds = load_beir(cfg.eval.dataset, cfg.eval.split)
    print(ds.summary())

    embedder = build_embedder(cfg.embed)
    print(f"backend={embedder.signature} dim={embedder.dim}")

    print("\n[1/3] corpus")
    doc_texts = [d.content for d in ds.documents]
    docs = embedder.encode_cached(doc_texts, tag=f"{cfg.eval.dataset}-corpus")
    print(f"  -> {docs.shape}")

    print("\n[2/3] test queries")
    q = embedder.encode_cached([x.text for x in ds.queries],
                               tag=f"{cfg.eval.dataset}-queries-{cfg.eval.split}")
    print(f"  -> {q.shape}")

    print("\n[3/3] train queries (for phase-kernel fitting)")
    pairs = load_train_pairs(cfg.eval.dataset)
    train_texts = sorted({p[0] for p in pairs})
    tq = embedder.encode_cached(train_texts, tag=f"{cfg.eval.dataset}-queries-train")
    print(f"  -> {tq.shape}  ({len(pairs)} query-positive pairs)")

    # Sanity check: normalised embeddings must give dot products in [-1, 1] and
    # a gold pair should out-score a random pair. If this fails, nothing
    # downstream is trustworthy.
    import numpy as np

    ids = {d: i for i, d in enumerate(ds.doc_ids)}
    qid0 = ds.queries[0].query_id
    gold = next(iter(ds.relevant(qid0)))
    sim_gold = float(q[0] @ docs[ids[gold]])
    rng = np.random.default_rng(0)
    sim_rand = float(np.mean([q[0] @ docs[i] for i in rng.integers(0, len(docs), 200)]))
    print(f"\nsanity: cos(query0, gold)={sim_gold:.4f} "
          f"vs mean cos(query0, random)={sim_rand:.4f}")
    assert -1.01 <= sim_gold <= 1.01, "embeddings are not normalised"
    assert sim_gold > sim_rand, "gold document does not out-score random documents"

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
