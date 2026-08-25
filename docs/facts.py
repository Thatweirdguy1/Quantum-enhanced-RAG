r"""Resolve ``{{token}}`` placeholders in document sources against the result files.

The paper, the slide deck and the diary all quote measured numbers. Typing those
numbers into prose is the failure this module exists to prevent: a figure typed into
four documents is a figure that can disagree with itself, and the reader who spots
the disagreement is the examiner. So the documents contain placeholders and the
numbers arrive from ``results/*.json`` at render time.

    "nDCG@10 was {{clean.qrag[full].ndcg@10}}"  ->  "nDCG@10 was 0.7140"

Two properties make this trustworthy rather than merely convenient:

* An unknown token is a hard error, not an empty string. A typo in a placeholder
  fails the build instead of shipping a document with ``{{...}}`` visible in it, or
  worse, a silently blank figure where a number should be.
* Nothing here computes a *new* quantity that the result files do not contain,
  except for explicitly-named derived facts (ratios, deltas, counts) whose formulas
  live in this file and nowhere else -- so a delta cannot be one value in the paper
  and another on a slide.

Tokens are dotted paths into the loaded JSON, with a few named shortcuts. Run

    python -m docs.facts

to print every available token with its resolved value, which is also the fastest
way to find the name of a number you want to quote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TOKEN = re.compile(r"\{\{([^{}]+)\}\}")


class FactError(KeyError):
    """A document asked for a number that the result files do not contain."""


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise FactError(f"{path.relative_to(ROOT)} is missing; run the experiment "
                        f"before building documents")
    return json.loads(path.read_text(encoding="utf8"))


# Identifiers and conventional dimensionless counts, not quantities. A seed printed
# as "20,260,720" cannot be pasted back into a command line, a thousands separator in
# a year or a hash is noise, and an embedding dimension is conventionally written
# 1024 rather than 1,024. A corpus size is a quantity and does take the separator.
BARE = {"seed", "config_hash", "python", "numpy", "platform", "dataset",
        "embed_dim"}


class Facts:
    """Read-only view over the result files, addressed by dotted token."""

    def __init__(self) -> None:
        self.exp = _load("experiment.json")
        self.kernel = _load("kernel_training.json")
        try:
            self.verification = _load("verification.json")
        except FactError:
            # Absent only before `python -m scripts.verify` has ever run. Documents
            # that quote it will fail loudly on the token, which is the intent.
            self.verification = {}
        self._named = self._build_named()

    # ------------------------------------------------------------------ shortcuts
    def _build_named(self) -> dict[str, Any]:
        exp, kt = self.exp, self.kernel
        clean = exp["clean"]
        base = clean["classical-baseline"]["metrics"]
        full = clean["qrag[full]"]["metrics"]
        poisoned = exp.get("poisoned", {})
        psys = poisoned.get("systems", {})
        man = poisoned.get("manifest", {})
        ver = self.verification

        named: dict[str, Any] = {
            # provenance
            "config_hash": exp["provenance"]["config_hash"],
            "seed": exp["provenance"]["seed"],
            "python": exp["provenance"]["python"],
            "numpy": exp["provenance"]["numpy"],
            "platform": exp["provenance"]["platform"],
            "n_docs": exp["dataset"]["n_docs"],
            "n_queries": exp["dataset"]["n_queries"],
            "n_judgments": exp["dataset"]["n_judgments"],
            "dataset": exp["dataset"]["name"],
            "n_systems": len(clean),

            # headline retrieval
            "base.recall@10": base["recall@10"],
            "base.ndcg@10": base["ndcg@10"],
            "base.mrr@10": base["mrr@10"],
            "full.recall@10": full["recall@10"],
            "full.ndcg@10": full["ndcg@10"],
            "full.mrr@10": full["mrr@10"],

            # kernel gates
            "gate_a.tau": kt["kernels"]["block-fidelity"]["divergence_after"]
                            ["kendall_tau_mean"],
            "gate_a.tau_global_before": kt["kernels"]["global-fidelity"]
                            ["divergence_before"]["kendall_tau_mean"],
            "gate_a.tau_theta0": kt["kernels"]["block-fidelity-theta0"]
                            ["divergence"]["kendall_tau_mean"],
            "gate_a.threshold": kt["gates"]["tau_ceiling"],
            "gate_a.pass": kt["gates"]["reorders"],
            "gate_b.mrr_before": kt["held_out_quality"]["baseline"]["mrr"],
            "gate_b.mrr_after": kt["held_out_quality"]["scorers"]
                            ["cosine + 0.25*block-fidelity"]["mrr"],
            "gate_b.top1_before": kt["held_out_quality"]["baseline"]["top1"],
            "gate_b.top1_after": kt["held_out_quality"]["scorers"]
                            ["cosine + 0.25*block-fidelity"]["top1"],
            "gate_b.delta_mrr": kt["gates"]["delta_mrr_fused"],
            "gate_b.delta_top1": kt["gates"]["delta_top1_fused"],
            "gate_b.pass": kt["gates"]["fusion_beats_cosine"],
            "n_train_pairs": kt["n_train_pairs"],
            "n_val_pairs": kt["held_out_quality"]["n_val_pairs"],
            "embed_dim": kt["embedding_dim"],
            "n_blocks": kt["kernels"]["block-fidelity"]["n_blocks"],
            "block_size": kt["kernels"]["block-fidelity"]["block_size"],
        }

        # verification counts
        if ver:
            named |= {
                "tests.passed": ver["tests"]["passed"],
                "tests.failed": ver["tests"]["failed"],
                "audit.passed": ver["audit"]["passed"],
                "audit.failed": ver["audit"]["failed"],
                "audit.warned": ver["audit"]["warned"],
                "audit.na": ver["audit"]["not_applicable"],
            }

        # significance: the count of cells that cross p<0.05, and the total. Both are
        # derived here so that "1 of 35" cannot become "2 of 35" in one document.
        cells = [c for sysdict in exp["significance"].values()
                 for c in sysdict.values()] if exp.get("significance") else []
        if cells and not isinstance(cells[0], dict):
            cells = []
        sig = [c for c in cells if c.get("significant")]
        named["sig.total"] = len(cells)
        named["sig.count"] = len(sig)
        if sig:
            best = max(sig, key=lambda c: abs(c.get("delta", 0.0)))
            named["sig.first.delta"] = best["delta"]
            named["sig.first.p"] = best["p_value"]

        # latency: slowdown of the full pipeline against its own baseline
        base_ms = base["latency_ms_mean"]
        full_ms = full["latency_ms_mean"]
        named |= {
            "base.ms": base_ms,
            "full.ms": full_ms,
            "base.run_seconds": clean["classical-baseline"]["run_seconds"],
            "full.run_seconds": clean["qrag[full]"]["run_seconds"],
            "slowdown": full_ms / base_ms if base_ms else float("nan"),
        }

        # quantum accounting, taken from the full pipeline
        q = clean["qrag[full]"].get("quantum", {})
        grover, qaoa = q.get("grover", {}), q.get("qaoa", {})
        if grover:
            named |= {
                "grover.oracle_queries": grover.get("mean_oracle_queries"),
                "grover.classical_queries":
                    grover.get("mean_classical_expected_queries"),
                "grover.reduction": grover.get("mean_query_reduction_factor"),
                "grover.success": grover.get("mean_success_probability"),
                "grover.qubits": grover.get("n_qubits"),
                "grover.candidates": grover.get("mean_candidates"),
                "grover.marked": grover.get("mean_marked"),
                "grover.overhead": grover.get("mean_simulation_overhead"),
                "grover.ms": grover.get("mean_wall_clock_ms"),
            }
        if qaoa:
            named |= {
                "qaoa.quality": qaoa.get("mean_solution_quality"),
                "qaoa.quality_worst": qaoa.get("min_solution_quality"),
                "qaoa.exact_rate": qaoa.get("fraction_optimal"),
                "qaoa.qubits": qaoa.get("n_qubits"),
                "qaoa.layers": qaoa.get("layers"),
                "qaoa.ms": qaoa.get("mean_wall_clock_ms"),
                "qaoa.optimiser_calls": qaoa.get("mean_optimiser_calls"),
                "qaoa.feasible_prob": qaoa.get("mean_feasible_probability"),
                "qaoa.max_gap": qaoa.get("max_objective_gap"),
                "qaoa.redundancy_before": qaoa.get("mean_redundancy_topk"),
                "qaoa.redundancy_after": qaoa.get("mean_redundancy_qaoa"),
                "qaoa.redundancy_drop": qaoa.get("mean_redundancy_reduction"),
            }

        # The Grover-only arm. Its simulation overhead is not the same number as the
        # full pipeline's, and quoting one where the other belongs is the kind of
        # slip that a shared token set is meant to make impossible.
        garm = clean.get("qrag[grover]", {}).get("quantum", {}).get("grover", {})
        if garm:
            named |= {
                "grover_arm.reduction": garm.get("mean_query_reduction_factor"),
                "grover_arm.overhead": garm.get("mean_simulation_overhead"),
                "grover_arm.success": garm.get("mean_success_probability"),
                "grover_arm.ms": garm.get("mean_wall_clock_ms"),
            }

        # the security result and its ablation control
        if psys:
            def occ(label: str):
                return psys.get(label, {}).get("attack", {}).get("context_occupancy")
            named |= {
                "poison.n_injected": man.get("n_injected"),
                "poison.n_targets": man.get("n_target_queries"),
                "poison.n_families": len(man.get("families", []) or []),
                "poison.occ_base": occ("classical-baseline"),
                "poison.occ_noqaoa": occ("qrag[no-qaoa]"),
                "poison.occ_full": occ("qrag[full]"),
                "poison.clean_rate_full":
                    psys.get("qrag[full]", {}).get("attack", {})
                        .get("clean_context_rate"),
                "poison.hit_rate_full":
                    psys.get("qrag[full]", {}).get("attack", {})
                        .get("top_k_hit_rate"),
                "poison.ndcg_full":
                    psys.get("qrag[full]", {}).get("metrics_poisoned", {})
                        .get("ndcg@10"),
                "poison.ndcg_drop_full":
                    psys.get("qrag[full]", {}).get("degradation_vs_clean", {})
                        .get("ndcg@10"),
            }
            no_q, fl = occ("qrag[no-qaoa]"), occ("qrag[full]")
            if no_q is not None and fl is not None:
                # The single number the security claim rests on. Defined once.
                named["poison.qaoa_effect"] = no_q - fl

            det = man.get("detector", {})
            if det:
                total = sum(v["n"] for v in det.values())
                flagged = sum(v["flagged"] for v in det.values())
                named |= {
                    "detector.total": total,
                    "detector.flagged": flagged,
                    "detector.rate": flagged / total if total else float("nan"),
                    "detector.families": len(det),
                }
                for family, stats in det.items():
                    named[f"detector.{family}"] = stats["detection_rate"]
        return named

    # -------------------------------------------------------------------- lookup
    def raw(self, token: str) -> Any:
        token = token.strip()
        if token in self._named:
            return self._named[token]
        # dotted path into experiment.json, e.g. clean.qrag[full].metrics.ndcg@10
        node: Any = self.exp
        for part in token.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise FactError(
                    f"unknown fact token {token!r} (failed at {part!r}). "
                    f"Run `python -m docs.facts` to list what is available.")
        return node

    def format(self, token: str) -> str:
        """Resolve a token to display text.

        Formatting is by magnitude and kind rather than per call site, so the same
        quantity is never shown to four decimals in one document and two in another.
        Explicit control is available with a ``|`` suffix: ``{{slowdown|1f}}``,
        ``{{qaoa.exact_rate|pct}}``, ``{{base.ms|0f}}``.
        """
        spec = None
        if "|" in token:
            token, spec = token.split("|", 1)
            spec = spec.strip()
        value = self.raw(token)

        if spec == "pct":
            return f"{value * 100:.1f}%"
        if spec and re.fullmatch(r"\d+f", spec):
            return f"{value:.{int(spec[:-1])}f}"
        if spec == "signed":
            return f"{value:+.4f}"
        if spec == "int":
            return f"{int(round(value)):,}"
        if spec == "bare":
            return str(value)

        if token in BARE:
            return str(value)
        if isinstance(value, bool):
            return "PASS" if value else "FAIL"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            if value != value:
                return "n/a"
            if abs(value) >= 100:
                return f"{value:,.0f}"
            if abs(value) >= 10:
                return f"{value:.1f}"
            return f"{value:.4f}"
        return str(value)

    def substitute(self, text: str) -> str:
        """Replace every ``{{token}}`` in *text*, collecting all failures at once."""
        missing: list[str] = []

        def repl(m: re.Match) -> str:
            try:
                return self.format(m.group(1))
            except FactError as exc:
                missing.append(str(exc))
                return m.group(0)

        out = TOKEN.sub(repl, text)
        if missing:
            raise FactError("unresolved fact tokens:\n  " + "\n  ".join(missing))
        return out

    def tokens(self) -> list[str]:
        return sorted(self._named)


def substitute(text: str) -> str:
    return Facts().substitute(text)


def main() -> int:
    f = Facts()
    width = max(len(t) for t in f.tokens())
    for token in f.tokens():
        print(f"{{{{{token}}}}}".ljust(width + 5), f.format(token))
    print(f"\n{len(f.tokens())} named tokens. Dotted paths into "
          f"results/experiment.json also resolve, "
          f"e.g. {{{{clean.qrag[full].metrics.ndcg@10}}}}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
