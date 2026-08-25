r"""The experiment of record. Every number in the report comes from here.

    python -m scripts.run_experiment                  # full: 300 queries, 7 systems
    python -m scripts.run_experiment --quick          # 40 queries, for a smoke test
    python -m scripts.run_experiment --no-poison      # clean arm only

Three arms, in order of what they are for:

1. **Ablation grid (clean corpus).** Each quantum stage switched on independently
   against the same classical baseline, so a difference is attributable to a
   component rather than to "the quantum pipeline". The baseline is a properly
   tuned hybrid -- when the kernel is off, its weight is redistributed onto cosine
   rather than dropped, so the baseline is not a strawman with a missing term.

2. **Quantum accounting.** Oracle queries, approximation ratio and feasible
   probability, reported separately from wall clock. A statevector simulator cannot
   win on latency and this pipeline does not claim it does: the Grover column is a
   query-complexity result, and the simulation overhead is printed next to it so the
   distinction cannot be lost in transcription.

3. **Poisoned corpus.** The security arm. Four adversarial families are injected,
   qrels untouched, and the same three pipelines are re-run. The hypothesis under
   test is specific: QAOA's redundancy penalty should suppress *clusters* of
   mutually-similar injected passages, so context occupancy should fall between
   ``qrag[no-qaoa]`` and ``qrag[full]``. If it does not, that is the finding.

Written to ``results/experiment.json`` after every system completes, so a run that
is interrupted still leaves usable partial results rather than nothing.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

from qrag.adversarial import (attack_report, detector_report, poison_corpus,
                              scan_ranked_context)
from qrag.config import CACHE_DIR, DEFAULT, RESULTS_DIR, Config
from qrag.data import Dataset, load_beir
from qrag.embed import build_embedder
from qrag.kernel import PhaseKernel
from qrag.metrics import EvalReport, compare_reports, format_comparison
from qrag.pipeline import BaselineRAG, QRAG, build_indexes
from qrag.provenance import snapshot as provenance_snapshot
from qrag.security import configure_logging

LOG = configure_logging()

# Each entry is (label, QRAG flags). ``None`` means the classical baseline, which
# is a different class rather than QRAG with everything disabled -- keeping them
# separate means the baseline cannot accidentally inherit a quantum stage.
ABLATIONS: tuple[tuple[str, dict | None], ...] = (
    ("classical-baseline", None),
    ("qrag[kernel]", dict(use_kernel=True, use_interference=False,
                          use_grover=False, use_qaoa=False)),
    ("qrag[kernel+interf]", dict(use_kernel=True, use_interference=True,
                                 use_grover=False, use_qaoa=False)),
    ("qrag[grover]", dict(use_kernel=False, use_interference=False,
                          use_grover=True, use_qaoa=False)),
    ("qrag[qaoa]", dict(use_kernel=False, use_interference=False,
                        use_grover=False, use_qaoa=True)),
    ("qrag[kernel+qaoa]", dict(use_kernel=True, use_interference=False,
                               use_grover=False, use_qaoa=True)),
    ("qrag[full]", dict(use_kernel=True, use_interference=True,
                        use_grover=True, use_qaoa=True)),
)

# The poisoned arm runs three systems only. qrag[no-qaoa] is the control that makes
# the security claim falsifiable: without it, any drop in context occupancy could
# be credited to the kernel instead of to the redundancy penalty.
POISON_SYSTEMS: tuple[tuple[str, dict | None], ...] = (
    ("classical-baseline", None),
    ("qrag[no-qaoa]", dict(use_kernel=True, use_interference=True,
                           use_grover=True, use_qaoa=False)),
    ("qrag[full]", dict(use_kernel=True, use_interference=True,
                        use_grover=True, use_qaoa=True)),
)


@dataclass
class Bench:
    """Everything a pipeline needs, built once and shared across systems.

    ``doc_vecs`` is carried alongside ``dense`` because DenseIndex does not expose
    its vectors -- it may be backed by faiss, where the array has been copied into
    the index. The poisoned arm needs the original rows to splice the injected ones
    in without re-embedding the clean corpus.
    """

    dataset: Dataset
    dense: object
    bm25: object
    qvecs: dict
    kernel: PhaseKernel
    cfg: Config
    doc_vecs: np.ndarray

    def system(self, label: str, flags: dict | None):
        if flags is None:
            return BaselineRAG(self.dataset, self.dense, self.bm25, self.qvecs, self.cfg)
        return QRAG(self.dataset, self.dense, self.bm25, self.qvecs, self.kernel,
                    self.cfg, label=label, **flags)


# ------------------------------------------------------------------ quantum trace
def _aggregate_traces(traces: list[dict]) -> dict:
    """Mean the per-query quantum diagnostics, keeping the accounting separable."""
    out: dict = {}
    grover = [t["grover"] for t in traces if "grover" in t]
    if grover:
        out["grover"] = {
            "n_queries": len(grover),
            "n_qubits": int(np.median([g["n_qubits"] for g in grover])),
            "mean_candidates": float(np.mean([g["n_candidates"] for g in grover])),
            "mean_marked": float(np.mean([g["n_marked"] for g in grover])),
            "mean_oracle_queries": float(np.mean([g["oracle_queries"] for g in grover])),
            "mean_classical_expected_queries":
                float(np.mean([g["classical_expected_queries"] for g in grover])),
            "mean_query_reduction_factor":
                float(np.mean([g["query_reduction_factor"] for g in grover])),
            "mean_success_probability":
                float(np.mean([g["success_probability"] for g in grover])),
            # Kept adjacent to the reduction factor on purpose: the first number is
            # a complexity result, the second is what simulating it actually costs.
            "mean_wall_clock_ms": float(np.mean([g["wall_clock_ms"] for g in grover])),
            "mean_simulation_overhead":
                float(np.mean([g["simulation_overhead"] for g in grover])),
        }
    qaoa = [t["qaoa"] for t in traces if "qaoa" in t]
    if qaoa:
        quality = np.array([q["solution_quality"] for q in qaoa
                            if q["solution_quality"] is not None], float)
        gaps = np.array([q["objective_gap"] for q in qaoa], float)
        red_q = np.array([q["redundancy_qaoa"] for q in qaoa], float)
        red_t = np.array([q["redundancy_topk"] for q in qaoa], float)
        out["qaoa"] = {
            "n_queries": len(qaoa),
            "n_qubits": int(np.median([q["n_qubits"] for q in qaoa])),
            "layers": int(np.median([q["layers"] for q in qaoa])),
            # Quality is in [0, 1] with 1 == the exact brute-force optimum over the
            # same exactly-k feasible set. fraction_optimal counts hitting it, and
            # mean_objective_gap says how far off the misses were in cost units.
            "mean_solution_quality": float(quality.mean()) if quality.size else None,
            "min_solution_quality": float(quality.min()) if quality.size else None,
            "n_degenerate": int(len(qaoa) - quality.size),
            "fraction_optimal": float(np.mean([q["is_optimal"] for q in qaoa])),
            "mean_objective_gap": float(gaps.mean()),
            "max_objective_gap": float(gaps.max()),
            "mean_feasible_probability":
                float(np.mean([q["feasible_probability"] for q in qaoa])),
            "mean_optimiser_calls": float(np.mean([q["optimiser_calls"] for q in qaoa])),
            "mean_wall_clock_ms": float(np.mean([q["wall_clock_ms"] for q in qaoa])),
            # The diversity claim, measured: redundancy of the QAOA-selected set
            # against the redundancy of taking the top-k by score alone.
            "mean_redundancy_qaoa": float(red_q.mean()),
            "mean_redundancy_topk": float(red_t.mean()),
            "mean_redundancy_reduction": float((red_t - red_q).mean()),
            "fraction_less_redundant": float(np.mean(red_q < red_t)),
        }
    kernel_modes = {t["kernel_mode"] for t in traces if "kernel_mode" in t}
    if kernel_modes:
        out["kernel_mode"] = sorted(kernel_modes)
    return out


def run_system(bench: Bench, label: str, flags: dict | None, *,
               k_values: tuple, top_k: int, verbose: bool = True) -> EvalReport:
    """Run one pipeline over every query and collect metrics plus traces."""
    pipeline = bench.system(label, flags)
    report = EvalReport(label, k_values)
    traces: list[dict] = []
    ranked_by_query: dict[str, list[str]] = {}

    t0 = time.perf_counter()
    for i, query in enumerate(bench.dataset.queries, start=1):
        result = pipeline.retrieve(query.query_id, query.text, top_k=top_k)
        report.add(query.query_id, result.ranked,
                   bench.dataset.qrels.get(query.query_id, {}),
                   latency_ms=result.latency_ms)
        traces.append(result.trace)
        ranked_by_query[query.query_id] = result.ranked
        if verbose and (i % 50 == 0 or i == len(bench.dataset.queries)):
            rate = i / (time.perf_counter() - t0)
            print(f"    {label:<22} {i:>4}/{len(bench.dataset.queries)}  "
                  f"{rate:.1f} q/s", flush=True)

    report.extras["quantum"] = _aggregate_traces(traces)
    report.extras["run_seconds"] = time.perf_counter() - t0
    report.extras["ranked"] = ranked_by_query
    return report


# ----------------------------------------------------------------- poisoned arm
def build_poisoned_bench(bench: Bench, embedder, *, n_targets: int,
                         per_family: int, budget: int) -> tuple[Bench, object]:
    """Inject adversarial passages, embed only the new ones, rebuild the indexes.

    Only the injected documents are re-embedded. The clean corpus vectors are
    reused unchanged, which keeps the two arms comparable: any difference in
    ranking comes from the new documents competing, not from a different encoder
    pass over the old ones.
    """
    print(f"  poisoning: {n_targets} target queries x {per_family} per family "
          f"x 4 families", flush=True)
    t0 = time.perf_counter()
    poisoned, manifest = poison_corpus(
        bench.dataset, n_targets=n_targets, per_query_per_family=per_family,
        seed=bench.cfg.eval.seed, embedder=embedder, query_vectors=bench.qvecs,
        optimise_budget=budget)
    print(f"  injected {manifest.n_injected} documents in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    manifest.detector = detector_report(
        [d for d in poisoned.documents if d.doc_id in manifest.injected])

    clean_ids = {d.doc_id for d in bench.dataset.documents}
    new_docs = [d for d in poisoned.documents if d.doc_id not in clean_ids]
    print(f"  embedding {len(new_docs)} injected passages", flush=True)
    new_vecs = embedder.encode([d.content for d in new_docs], show_progress=True)

    # Row order must match poisoned.doc_ids exactly, so index the clean vectors by
    # position and append the new ones in the order poison_corpus emitted them.
    clean_pos = {d.doc_id: i for i, d in enumerate(bench.dataset.documents)}
    new_pos = {d.doc_id: i for i, d in enumerate(new_docs)}
    rows = [bench.doc_vecs[clean_pos[d]] if d in clean_pos
            else new_vecs[new_pos[d]] for d in poisoned.doc_ids]
    vectors = np.stack(rows)

    dense, bm25 = build_indexes(poisoned, vectors, use_faiss=True)
    return Bench(poisoned, dense, bm25, bench.qvecs, bench.kernel, bench.cfg,
                 vectors), manifest


def run_poison_arm(bench: Bench, embedder, args, clean_means: dict) -> dict:
    poisoned_bench, manifest = build_poisoned_bench(
        bench, embedder, n_targets=args.poison_targets,
        per_family=args.poison_per_family, budget=args.poison_budget)

    out: dict = {"manifest": manifest.as_dict(), "systems": {}}
    for label, flags in POISON_SYSTEMS:
        print(f"  [poisoned] {label}", flush=True)
        report = run_system(poisoned_bench, label, flags,
                            k_values=bench.cfg.eval.k_values, top_k=args.top_k)
        ranked = report.extras.pop("ranked")
        attack = attack_report(ranked, manifest,
                              context_k=bench.cfg.gen.max_context_docs,
                              top_k=bench.cfg.eval.primary_k)
        context_scan = scan_ranked_context(ranked, poisoned_bench.dataset,
                                           context_k=bench.cfg.gen.max_context_docs)
        means = report.means()
        out["systems"][label] = {
            "metrics_poisoned": means,
            # Retrieval quality on the poisoned corpus vs the clean run. The qrels
            # are identical, so this difference is the cost the attack imposes on
            # ordinary queries -- not just on the attacked ones.
            "degradation_vs_clean": {
                m: means[m] - clean_means.get(label, {}).get(m, float("nan"))
                for m in ("recall@10", "ndcg@10", "mrr@10")
                if m in means},
            "attack": attack,
            "context_detector": context_scan,
            "quantum": report.extras.get("quantum", {}),
        }
        _checkpoint({"poisoned_partial": out},
                    args.out.replace(".json", "_poison.json"), bench.cfg)
    return out


# --------------------------------------------------------------------- reporting
def _checkpoint(payload: dict, path: str, cfg: Config | None = None) -> None:
    """Write an intermediate or final result file.

    Every result file carries its provenance, including the partial ones. A
    checkpoint that records metrics but not the configuration that produced them is
    a file whose numbers cannot be traced back to a run, and ``security_audit.py``
    check PROD-3 flags exactly that -- it caught the poisoned-arm checkpoint being
    written bare, which is what this parameter fixes.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg is not None and "provenance" not in payload:
        payload = {"provenance": _provenance(cfg), "config": cfg.to_dict(),
                   "partial": True, **payload}
    (RESULTS_DIR / path).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf8")


def _provenance(cfg: Config) -> dict:
    # Code and platform identity is shared with the other result writers; the config
    # hash and seed are what make an *experiment* result reproducible on top of it.
    return {"config_hash": cfg.hash(), **provenance_snapshot(), "seed": cfg.eval.seed}


def print_summary(reports: dict[str, EvalReport], comparisons: dict) -> None:
    print("\n" + "=" * 92)
    print("ABLATION GRID -- clean corpus")
    print("=" * 92)
    for report in reports.values():
        print(report.table(["recall@10", "ndcg@10", "mrr@10", "recall@5"]))

    print("\n" + "=" * 92)
    print("PAIRED BOOTSTRAP vs classical-baseline (2000 resamples, seed fixed)")
    print("=" * 92)
    for label, comparison in comparisons.items():
        print(f"\n{label}")
        print(format_comparison(comparison, "baseline", label))

    print("\n" + "=" * 92)
    print("QUANTUM ACCOUNTING -- complexity and simulation cost are separate columns")
    print("=" * 92)
    for label, report in reports.items():
        q = report.extras.get("quantum", {})
        if "grover" in q:
            g = q["grover"]
            print(f"{label:<22} grover: {g['n_qubits']}q  "
                  f"oracle_queries={g['mean_oracle_queries']:.2f}  "
                  f"classical_expected={g['mean_classical_expected_queries']:.2f}  "
                  f"reduction={g['mean_query_reduction_factor']:.2f}x  "
                  f"P(success)={g['mean_success_probability']:.3f}  "
                  f"[sim {g['mean_wall_clock_ms']:.2f}ms, "
                  f"overhead {g['mean_simulation_overhead']:.2f}x]")
        if "qaoa" in q:
            a = q["qaoa"]
            qual = ("n/a" if a["mean_solution_quality"] is None
                    else f"{a['mean_solution_quality']:.4f} "
                         f"(min {a['min_solution_quality']:.3f})")
            print(f"{label:<22} qaoa:   {a['n_qubits']}q p={a['layers']}  "
                  f"quality={qual}  "
                  f"optimal on {a['fraction_optimal']*100:.1f}%  "
                  f"gap<={a['max_objective_gap']:.4f}  "
                  f"P(feasible)={a['mean_feasible_probability']:.3f}  "
                  f"redundancy {a['mean_redundancy_topk']:.4f}->"
                  f"{a['mean_redundancy_qaoa']:.4f} "
                  f"(lower on {a['fraction_less_redundant']*100:.1f}%)  "
                  f"[sim {a['mean_wall_clock_ms']:.0f}ms]")


def print_poison_summary(poison: dict) -> None:
    print("\n" + "=" * 92)
    print("POISONED CORPUS -- context occupancy is the attacker's objective")
    print("=" * 92)
    man = poison["manifest"]
    print(f"injected {man['n_injected']} passages across {man['n_target_queries']} "
          f"queries; families: {', '.join(man['families'])}")
    det = man.get("detector", {})
    if det:
        # detector_report returns {family: {n, flagged, by_severity, detection_rate}}.
        # An earlier version of this printer looked for 'flagged_fraction' and
        # 'by_family', found neither, and so reported "0.0% flagged overall" while
        # silently printing no per-family lines at all -- with the real rate being
        # 100/400. That is the precise failure SECURITY.md forbids: an aggregate that
        # hides which families the detector misses. Derive both from the real shape.
        total = sum(v["n"] for v in det.values())
        flagged = sum(v["flagged"] for v in det.values())
        print(f"pattern detector on injected text: "
              f"{flagged}/{total} = {flagged / max(total, 1) * 100:.1f}% flagged "
              f"overall -- read the per-family rows, not this number")
        for family, stats in det.items():
            print(f"    {family:<24} {stats['detection_rate'] * 100:>5.1f}% flagged "
                  f"({stats['flagged']}/{stats['n']})")
    print(f"\n{'system':<22}{'ctx occupancy':>15}{'clean ctx':>12}"
          f"{'top10 hit':>12}{'1st adv rank':>14}{'ndcg@10':>10}")
    print("-" * 92)
    for label, block in poison["systems"].items():
        a, m = block["attack"], block["metrics_poisoned"]
        rank = a["median_first_adv_rank"]
        print(f"{label:<22}{a['context_occupancy']:>15.4f}"
              f"{a['clean_context_rate']:>12.4f}{a['top_k_hit_rate']:>12.4f}"
              f"{(f'{rank:.1f}' if rank else '-'):>14}{m.get('ndcg@10', 0):>10.4f}")
    print("\nper-family share of occupied context slots")
    for label, block in poison["systems"].items():
        share = block["attack"]["context_share_by_family"]
        print(f"  {label:<22}" + "  ".join(f"{k}={v:.3f}" for k, v in share.items()))


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queries", type=int, default=None,
                    help="limit query count (default: all judged queries)")
    ap.add_argument("--quick", action="store_true",
                    help="40 queries, 3 systems, small poison arm")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--no-poison", action="store_true")
    ap.add_argument("--poison-targets", type=int, default=50)
    ap.add_argument("--poison-per-family", type=int, default=2)
    ap.add_argument("--poison-budget", type=int, default=24)
    ap.add_argument("--out", default="experiment.json")
    args = ap.parse_args()

    cfg = DEFAULT
    ablations = ABLATIONS
    if args.quick:
        args.queries = args.queries or 40
        args.poison_targets = min(args.poison_targets, 12)
        ablations = tuple(a for a in ABLATIONS if a[0] in
                          ("classical-baseline", "qrag[kernel+interf]", "qrag[full]"))

    print("=" * 92)
    print(f"Q-RAG experiment  config={cfg.hash()}  "
          f"{'QUICK' if args.quick else 'FULL'} run")
    print("=" * 92)

    dataset = load_beir(cfg.eval.dataset, cfg.eval.split, n_queries=args.queries)
    print(dataset.summary(), flush=True)

    embedder = build_embedder(cfg.embed)
    doc_vecs = embedder.encode_cached([d.content for d in dataset.documents],
                                      f"{cfg.eval.dataset}-corpus")
    query_vecs = embedder.encode_cached([q.text for q in dataset.queries],
                                        f"{cfg.eval.dataset}-queries-{cfg.eval.split}")
    qvecs = {q.query_id: query_vecs[i] for i, q in enumerate(dataset.queries)}
    dense, bm25 = build_indexes(dataset, doc_vecs)

    kernel_path = CACHE_DIR / "phase_kernel.npz"
    if not kernel_path.exists():
        print(f"error: {kernel_path} not found. Run:\n"
              "  python -m scripts.train_kernel", file=sys.stderr)
        return 2
    kernel = PhaseKernel.load(kernel_path, cfg.kernel)
    if not kernel.trained:
        print("error: kernel file exists but is untrained; the kernel leg would be "
              "a no-op and the comparison meaningless.", file=sys.stderr)
        return 2
    print(f"kernel: {kernel.kind}, trained, {kernel.param_summary()}", flush=True)

    bench = Bench(dataset, dense, bm25, qvecs, kernel, cfg, doc_vecs)

    # ---------------------------------------------------------------- clean arm
    print("\nrunning ablation grid", flush=True)
    reports: dict[str, EvalReport] = {}
    payload: dict = {
        "provenance": _provenance(cfg),
        "config": cfg.to_dict(),
        "dataset": {"name": dataset.name, "n_docs": len(dataset.documents),
                    "n_queries": len(dataset.queries),
                    "n_judgments": sum(len(v) for v in dataset.qrels.values())},
        "kernel": kernel.param_summary(),
        "clean": {},
    }
    for label, flags in ablations:
        report = run_system(bench, label, flags,
                            k_values=cfg.eval.k_values, top_k=args.top_k)
        report.extras.pop("ranked", None)
        reports[label] = report
        payload["clean"][label] = {
            "metrics": report.means(),
            "per_query": report.per_query,
            "quantum": report.extras.get("quantum", {}),
            "run_seconds": report.extras["run_seconds"],
        }
        _checkpoint(payload, args.out)

    baseline = reports["classical-baseline"]
    comparisons = {label: compare_reports(baseline, report,
                                          n_samples=cfg.eval.bootstrap_samples,
                                          seed=cfg.eval.seed)
                   for label, report in reports.items()
                   if label != "classical-baseline"}
    payload["significance"] = comparisons
    _checkpoint(payload, args.out)
    print_summary(reports, comparisons)

    # -------------------------------------------------------------- poisoned arm
    if not args.no_poison:
        print("\nrunning poisoned-corpus arm", flush=True)
        clean_means = {label: r.means() for label, r in reports.items()}
        poison = run_poison_arm(bench, embedder, args, clean_means)
        payload["poisoned"] = poison
        _checkpoint(payload, args.out)
        print_poison_summary(poison)

    print(f"\nwrote results/{args.out}")
    print("Every table in the report is generated from this file; regenerate with "
          "the same command and the same seed to reproduce it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
