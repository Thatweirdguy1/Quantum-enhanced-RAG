"""Dataset loading for BEIR-format corpora.

SciFact is used as the primary benchmark: 5,183 abstracts, 300 test queries with
public relevance judgments, and 919 train judgments that provide the supervision
needed to fit the phase kernel. It is small enough for exact search, which means
the baseline and Q-RAG are compared against the same ground truth rather than
against an approximate index's idea of it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    # Marks documents injected by the adversarial suite. None for corpus docs.
    adversarial_kind: str | None = None

    @property
    def content(self) -> str:
        """Title and body concatenated, as is conventional for BEIR scoring."""
        if self.title and self.text:
            return f"{self.title}. {self.text}"
        return self.title or self.text


@dataclass
class Query:
    query_id: str
    text: str


@dataclass
class Dataset:
    name: str
    documents: list[Document]
    queries: list[Query]
    # qrels[query_id][doc_id] = graded relevance (SciFact is binary: 1)
    qrels: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        self._by_id = {d.doc_id: d for d in self.documents}

    def doc(self, doc_id: str) -> Document:
        return self._by_id[doc_id]

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]

    def relevant(self, query_id: str) -> set[str]:
        return {d for d, s in self.qrels.get(query_id, {}).items() if s > 0}

    def summary(self) -> str:
        n_rel = sum(len(v) for v in self.qrels.values())
        adv = sum(1 for d in self.documents if d.adversarial_kind)
        return (
            f"{self.name}: {len(self.documents)} docs "
            f"({adv} adversarial), {len(self.queries)} queries, "
            f"{n_rel} judgments"
        )


def _read_jsonl(path: Path):
    with path.open(encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf8") as fh:
        header = fh.readline()  # query-id \t corpus-id \t score
        if "query-id" not in header:
            fh.seek(0)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            qrels[qid][did] = int(float(score))
    return dict(qrels)


def load_beir(name: str = "scifact", split: str = "test",
              n_queries: int | None = None, root: Path | None = None) -> Dataset:
    """Load a BEIR dataset from ``data/<name>/``.

    Only queries that carry at least one judgment in ``split`` are kept: an
    unjudged query contributes nothing to Recall/MRR/nDCG but would silently
    drag every mean toward zero.
    """
    base = (root or DATA_DIR) / name
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Fetch it with:\n"
            f'  curl -sSL -o data/raw/{name}.zip '
            f'"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"'
        )

    documents = [
        Document(doc_id=str(r["_id"]), title=r.get("title", ""), text=r.get("text", ""))
        for r in _read_jsonl(base / "corpus.jsonl")
    ]
    qrels = load_qrels(base / "qrels" / f"{split}.tsv")

    judged = set(qrels)
    queries = [
        Query(query_id=str(r["_id"]), text=r["text"])
        for r in _read_jsonl(base / "queries.jsonl")
        if str(r["_id"]) in judged
    ]
    queries.sort(key=lambda q: int(q.query_id) if q.query_id.isdigit() else q.query_id)
    if n_queries is not None:
        queries = queries[:n_queries]
        keep = {q.query_id for q in queries}
        qrels = {k: v for k, v in qrels.items() if k in keep}

    return Dataset(name=f"{name}/{split}", documents=documents, queries=queries, qrels=qrels)


def load_train_pairs(name: str = "scifact", root: Path | None = None
                     ) -> list[tuple[str, str]]:
    """Return (query_text, positive_doc_id) pairs from the *train* qrels.

    Used to fit the phase kernel. The test split is never touched during
    training -- that separation is what makes the week 9 comparison meaningful.
    """
    base = (root or DATA_DIR) / name
    qrels = load_qrels(base / "qrels" / "train.tsv")
    qtext = {str(r["_id"]): r["text"] for r in _read_jsonl(base / "queries.jsonl")}
    pairs = []
    for qid, docs in qrels.items():
        if qid not in qtext:
            continue
        for did, score in docs.items():
            if score > 0:
                pairs.append((qtext[qid], str(did)))
    return pairs


if __name__ == "__main__":  # quick sanity check
    ds = load_beir()
    print(ds.summary())
    print("train pairs:", len(load_train_pairs()))
