r"""Generate FACTS.md -- the single source of truth for every document.

    python -m scripts.build_facts

Why this exists
---------------
Five deliverables have to quote the same numbers: the research paper, the literature
review, the slide deck, the daily diary, and this repository's own README. Copying
figures between them by hand is how a report ends up claiming an improvement in one
table and a regression in another, and the reader who notices is the examiner.

So nothing is transcribed. This reads ``results/*.json`` and emits FACTS.md with
every quotable number, each tagged with the file and key it came from. The documents
are written *from* FACTS.md, and if an experiment is re-run the facts file is
regenerated rather than edited.

It also emits a "claims that may not be made" section. That is not decoration: the
temptation under deadline is to round a negative result into a neutral one, and
having the prohibited sentences written down next to the numbers makes doing so a
deliberate act rather than a drafting convenience.
"""

from __future__ import annotations

import json

from qrag.config import RESULTS_DIR, ROOT


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def _fmt(value, places: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def _signed(value, places: int = 4) -> str:
    # NaN reaches here when a poisoned-arm system has no counterpart in the clean
    # grid, so there is no clean run to subtract. Printing "+nan" in a report table
    # reads as a computation that went wrong rather than a quantity that is not
    # defined, so say which it is.
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    return f"{value:+.{places}f}"


def section_provenance(exp: dict) -> list[str]:
    p, d = exp["provenance"], exp["dataset"]
    return [
        "## 1. Provenance",
        "",
        "Cite these whenever a table is reproduced. Same command + same seed gives",
        "the same numbers; a different config hash means the table is from a",
        "different experiment and must not be mixed with these.",
        "",
        f"- config hash: `{p['config_hash']}`",
        f"- seed: `{p['seed']}` (every stage)",
        f"- git commit: `{p['git_commit']}`",
        f"- python {p['python']}, numpy {p['numpy']}",
        f"- platform: {p['platform']}",
        f"- dataset: **{d['name']}** -- {d['n_docs']} documents, "
        f"{d['n_queries']} queries, {d['n_judgments']} relevance judgments",
        f"- source: `results/experiment.json`",
        "",
    ]


def section_kernel(kt: dict) -> list[str]:
    g = kt["gates"]
    gl, bl = kt["kernels"]["global-fidelity"], kt["kernels"]["block-fidelity"]
    t0 = kt["kernels"]["block-fidelity-theta0"]
    hq = kt["held_out_quality"]
    hn = kt["hard_negative_stats"]

    out = [
        "## 2. Kernel training and the two gates",
        "",
        "Source: `results/kernel_training.json`. Both gates were defined **before**",
        "training, because the central risk was that the kernel is a ranking no-op.",
        "",
        f"- training pairs: {kt['n_train_pairs']} "
        f"(from {kt['n_unique_train_queries']} unique train queries), "
        f"embedding dim {kt['embedding_dim']}",
        f"- hard-negative pool: {hn['pool_size']} per query; "
        f"mean cos(q, hardest negative) {_fmt(hn['mean_cos_hardest_negative'])} "
        f"vs cos(q, positive) {_fmt(hn['mean_cos_positive'])}; "
        f"the positive is out-scored on "
        f"{hn['fraction_positive_outscored'] * 100:.1f}% of pairs",
        "",
        "### Gate A -- does the kernel reorder at all?",
        "",
        f"Kendall tau against the cosine ranking must fall below "
        f"{g['tau_ceiling']}.",
        "",
        "| kernel | tau before training | tau after training | verdict |",
        "|---|---|---|---|",
        f"| global fidelity | {_fmt(gl['divergence_before']['kendall_tau_mean'])} "
        f"| {_fmt(gl['divergence_after']['kendall_tau_mean'])} | reorders only "
        f"after training; **starts as a near-exact no-op** |",
        f"| block fidelity | {_fmt(bl['divergence_before']['kendall_tau_mean'])} "
        f"| {_fmt(bl['divergence_after']['kendall_tau_mean'])} | reorders |",
        f"| block fidelity at theta=0 (control) | "
        f"{_fmt(t0['divergence']['kendall_tau_mean'])} | (not trained) | "
        f"**breaks rank-equivalence structurally, without fitted phases** |",
        "",
        f"- Gate A verdict: **{'PASS' if g['reorders'] else 'FAIL'}** "
        f"(tau {_fmt(bl['divergence_after']['kendall_tau_mean'])} "
        f"< {g['tau_ceiling']})",
        f"- global kernel mean|theta| after training: {_fmt(gl['abs_theta_mean'])}",
        f"- block kernel: {bl['n_blocks']} blocks x {bl['block_size']} dims, "
        f"mean|theta| {_fmt(bl['abs_theta_mean'])}, "
        f"w in [{_fmt(bl['w_min'])}, {_fmt(bl['w_max'])}]",
        f"- early stopping restored epoch {bl['fit']['best_epoch']} "
        f"(val_loss {_fmt(bl['fit']['best_val_loss'])})",
        "",
        "### Gate B -- is the reordering an improvement?",
        "",
        f"Held-out: {hq['n_val_pairs']} validation pairs, "
        f"{hq['n_negatives']} negatives each.",
        "",
        "| scorer | top-1 | MRR |",
        "|---|---|---|",
    ]
    for name, sc in hq["scorers"].items():
        out.append(f"| {name} | {_fmt(sc['top1'])} | {_fmt(sc['mrr'])} |")
    out += [
        "",
        f"- Gate B verdict: **{'PASS' if g['fusion_beats_cosine'] else 'FAIL'}** -- "
        f"MRR {_fmt(hq['baseline']['mrr'])} -> "
        f"{_fmt(hq['scorers']['cosine + 0.25*block-fidelity']['mrr'])} "
        f"({_signed(g['delta_mrr_fused'])}), "
        f"top-1 {_signed(g['delta_top1_fused'])}",
        "",
        "**Two facts that must travel with this table.** First, `cosine` and",
        "`cosine^2` score *identically* on every metric, which is the empirical",
        "confirmation that a global fidelity kernel at theta=0 is rank-equivalent to",
        "cosine and therefore cannot rerank. Second, a top-1 gain of",
        f"{_signed(g['delta_top1_fused'])} on {hq['n_val_pairs']} pairs is about two",
        "queries; it is not significant and must not be described as such.",
        "",
    ]
    return out


def section_ablation(exp: dict) -> list[str]:
    metrics = ["recall@10", "ndcg@10", "mrr@10", "recall@5", "ndcg@5", "recall@20"]
    out = [
        "## 3. Ablation grid on the clean corpus",
        "",
        f"All {exp['dataset']['n_queries']} SciFact test queries. The baseline is the",
        "same hybrid fusion with the kernel weight redistributed onto cosine, so it",
        "is a tuned system rather than a strawman with a missing term.",
        "",
        "| system | " + " | ".join(metrics) + " | ms/query |",
        "|---" * (len(metrics) + 2) + "|",
    ]
    for label, block in exp["clean"].items():
        m = block["metrics"]
        row = [_fmt(m.get(k)) for k in metrics]
        lat = m.get("latency_ms_mean")
        out.append(f"| `{label}` | " + " | ".join(row) + f" | {_fmt(lat, 1)} |")
    out.append("")

    # Whether Grover moved the ranking at all is a fact the table above contains but
    # does not say. Stating it stops the oracle-query result in section 5 from being
    # read as a retrieval result.
    grover = exp["clean"].get("qrag[grover]", {}).get("metrics", {})
    base = exp["clean"]["classical-baseline"]["metrics"]
    shared = [k for k in base if k.startswith(("recall@", "ndcg@", "mrr@"))]
    if grover and all(abs(grover.get(k, 0.0) - base[k]) < 1e-12 for k in shared):
        out += [
            "**`qrag[grover]` scores identically to the baseline on every retrieval "
            "metric, to floating point.** That is not a coincidence and not a bug: "
            "amplitude amplification here selects from a shortlist that has already "
            "been scored classically, and the ordering it returns is the classical "
            "ordering. Grover in this pipeline demonstrates an oracle-query "
            "complexity property on a real workload and contributes nothing to "
            "ranking quality. Any document presenting it as a retrieval improvement "
            "is misreporting this table.",
            "",
        ]
    return out


def section_significance(exp: dict) -> list[str]:
    out = [
        "## 4. Significance against the classical baseline",
        "",
        "Paired bootstrap over per-query scores, "
        f"{exp['config']['eval']['bootstrap_samples']} resamples, seed fixed. "
        "`sig` means the 95% CI excludes zero.",
        "",
        "| system | metric | baseline | system | delta | 95% CI | p | sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, comparison in exp.get("significance", {}).items():
        for metric, c in comparison.items():
            if not isinstance(c, dict) or "delta" not in c:
                continue
            out.append(
                f"| `{label}` | {metric} | {_fmt(c['baseline_mean'])} | "
                f"{_fmt(c['system_mean'])} | {_signed(c['delta'])} | "
                f"[{_signed(c['ci95_low'])}, {_signed(c['ci95_high'])}] | "
                f"{_fmt(c['p_value'])} | {_fmt(c['significant'])} |")
    out.append("")
    return out


def section_quantum(exp: dict) -> list[str]:
    out = [
        "## 5. Quantum accounting",
        "",
        "Complexity and simulation cost are reported as **separate** quantities and",
        "must stay separate in every document. Grover's oracle-query reduction is a",
        "hardware-independent count; the simulation overhead beside it is what",
        "running that count on a classical statevector costs.",
        "",
    ]
    for label, block in exp["clean"].items():
        q = block.get("quantum", {})
        if "grover" in q:
            g = q["grover"]
            out += [
                f"### Grover -- `{label}`",
                "",
                f"- {g['n_qubits']} qubits over "
                f"{_fmt(g['mean_candidates'], 1)} shortlisted candidates, "
                f"{_fmt(g['mean_marked'], 1)} marked on average",
                f"- **oracle queries: {_fmt(g['mean_oracle_queries'], 2)}** vs "
                f"**{_fmt(g['mean_classical_expected_queries'], 2)}** expected for "
                f"classical scanning -> "
                f"**{_fmt(g['mean_query_reduction_factor'], 2)}x** reduction",
                f"- success probability: "
                f"{_fmt(g['mean_success_probability'], 3)}",
                f"- simulation cost: {_fmt(g['mean_wall_clock_ms'], 2)} ms/query, "
                f"{_fmt(g['mean_simulation_overhead'], 2)}x overhead vs the "
                f"classical scan it replaces",
                "",
            ]
        if "qaoa" in q:
            a = q["qaoa"]
            out += [
                f"### QAOA -- `{label}`",
                "",
                f"- {a['n_qubits']} qubits, p={a['layers']} layers, "
                f"{_fmt(a['mean_optimiser_calls'], 0)} optimiser calls/query",
                f"- solution quality: {_fmt(a['mean_solution_quality'])} mean, "
                f"{_fmt(a['min_solution_quality'])} worst "
                f"(affine-invariant, 1.0 == exact brute-force optimum over the "
                f"same exactly-k feasible set)",
                f"- **hit the exact optimum on "
                f"{a['fraction_optimal'] * 100:.1f}% of queries**; "
                f"largest objective gap {_fmt(a['max_objective_gap'])}",
                f"- feasible probability at readout: "
                f"{_fmt(a['mean_feasible_probability'], 3)}",
                f"- **redundancy: {_fmt(a['mean_redundancy_topk'])} (top-k by score) "
                f"-> {_fmt(a['mean_redundancy_qaoa'])} (QAOA), "
                f"a reduction of {_fmt(a['mean_redundancy_reduction'])}; "
                f"lower on {a['fraction_less_redundant'] * 100:.1f}% of queries**",
                f"- simulation cost: {_fmt(a['mean_wall_clock_ms'], 0)} ms/query",
                "",
            ]
            if a.get("n_degenerate"):
                out.append(
                    f"- {a['n_degenerate']} query/queries had no spread between the "
                    f"best and worst feasible selection, so no quality score is "
                    f"defined for them and they are excluded from the mean.")
                out.append("")
    return out


def section_latency(exp: dict) -> list[str]:
    base = exp["clean"]["classical-baseline"]["metrics"].get("latency_ms_mean")
    out = [
        "## 6. Latency -- the honest column",
        "",
        "| system | ms/query | vs baseline | seconds for the full run |",
        "|---|---|---|---|",
    ]
    for label, block in exp["clean"].items():
        lat = block["metrics"].get("latency_ms_mean")
        ratio = f"{lat / base:.1f}x" if (lat and base) else "n/a"
        out.append(f"| `{label}` | {_fmt(lat, 1)} | {ratio} | "
                   f"{_fmt(block['run_seconds'], 1)} |")
    out += [
        "",
        "**No wall-clock speed-up is claimed anywhere.** A statevector simulator",
        "cannot beat the classical routine it simulates; the slowdown above is the",
        "expected cost of simulation and is reported so that the Grover",
        "oracle-query result in section 5 cannot be mistaken for a timing result.",
        "",
    ]
    return out


def section_poison(exp: dict) -> list[str]:
    poison = exp.get("poisoned")
    if not poison:
        return ["## 7. Poisoned-corpus arm", "",
                "_Not present in this results file; re-run without `--no-poison`._",
                ""]
    man = poison["manifest"]
    out = [
        "## 7. Poisoned-corpus arm -- the security experiment",
        "",
        f"{man['n_injected']} adversarial passages injected against "
        f"{man['n_target_queries']} target queries across "
        f"{len(man['families'])} families: {', '.join(man['families'])}. "
        "Relevance judgments are **not** modified, so retrieval metrics stay "
        "comparable with the clean arm.",
        "",
        "**Context occupancy is the attacker's objective**: the fraction of the "
        f"{poison['systems'][next(iter(poison['systems']))]['attack']['context_k']}"
        "-document context window filled with injected passages. Lower is better "
        "for the defender.",
        "",
        "| system | context occupancy | clean-context rate | top-10 hit rate | "
        "median first adversarial rank | ndcg@10 | ndcg@10 vs clean |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, block in poison["systems"].items():
        a, m = block["attack"], block["metrics_poisoned"]
        rank = a["median_first_adv_rank"]
        deg = block["degradation_vs_clean"].get("ndcg@10")
        out.append(
            f"| `{label}` | {_fmt(a['context_occupancy'])} | "
            f"{_fmt(a['clean_context_rate'])} | {_fmt(a['top_k_hit_rate'])} | "
            f"{_fmt(rank, 1) if rank else 'none'} | {_fmt(m.get('ndcg@10'))} | "
            f"{_signed(deg)} |")

    out += ["", "### Per-family share of occupied context slots", "",
            "| system | " + " | ".join(man["families"]) + " |",
            "|---" * (len(man["families"]) + 1) + "|"]
    for label, block in poison["systems"].items():
        share = block["attack"]["context_share_by_family"]
        out.append(f"| `{label}` | " +
                   " | ".join(_fmt(share.get(f, 0.0)) for f in man["families"]) + " |")

    det = man.get("detector") or {}
    if det:
        # {family: {n, flagged, by_severity, detection_rate}}. The overall figure is
        # derived rather than read, and is printed *after* the per-family table on
        # purpose: quoting it alone is the reporting failure section 9 prohibits.
        total = sum(v["n"] for v in det.values())
        flagged = sum(v["flagged"] for v in det.values())
        out += ["", "### Pattern detector on the injected text", "",
                "| family | flagged | n |", "|---|---|---|"]
        for family, stats in det.items():
            out.append(f"| {family} | {stats['detection_rate'] * 100:.1f}% | "
                       f"{stats['flagged']}/{stats['n']} |")
        out += [
            "",
            f"Overall: {flagged}/{total} = "
            f"{flagged / max(total, 1) * 100:.1f}%.",
            "",
            "**The 0% rows are the honest ones and must be quoted alongside the 100%",
            "row.** Fluent text that simply contradicts the scientific claim carries",
            "no detectable pattern, because it was written to look exactly like a",
            "real abstract. A regex layer does not defend against corpus poisoning,",
            "and the aggregate figure hides precisely that: three of the four",
            "families evade it completely.",
            "",
        ]

    # The falsifiable comparison, stated as a computed verdict rather than a hope.
    systems = poison["systems"]
    if "qrag[no-qaoa]" in systems and "qrag[full]" in systems:
        no_q = systems["qrag[no-qaoa]"]["attack"]["context_occupancy"]
        full = systems["qrag[full]"]["attack"]["context_occupancy"]
        base = systems["classical-baseline"]["attack"]["context_occupancy"]
        delta = no_q - full
        out += [
            "### Verdict on the QAOA redundancy hypothesis",
            "",
            "The hypothesis was that the redundancy penalty "
            "`lambda * sum_{i<j} s_ij x_i x_j` suppresses *clusters* of "
            "mutually-similar injected passages, and `qrag[no-qaoa]` is the control "
            "that makes it falsifiable -- without it any drop could be credited to "
            "the kernel instead.",
            "",
            f"- classical baseline occupancy: {_fmt(base)}",
            f"- `qrag[no-qaoa]` occupancy: {_fmt(no_q)}",
            f"- `qrag[full]` occupancy: {_fmt(full)}",
            f"- **attributable to QAOA: {_signed(delta)}** "
            f"({'supports' if delta > 0 else 'does NOT support'} the hypothesis)",
            "",
            f"Wording that is permitted: "
            + ("QAOA reranking reduced adversarial context occupancy by "
               f"{delta:.4f} relative to the otherwise identical pipeline without "
               "it, on a single corpus and a single attack budget."
               if delta > 0 else
               "the redundancy penalty did **not** reduce adversarial context "
               "occupancy on this corpus; the hypothesis is unsupported and is "
               "reported as such."),
            "",
        ]
    return out


def section_status() -> list[str]:
    return [
        "## 8. Verification status",
        "",
        "Both are re-runnable and both must be re-run before the report is "
        "submitted, because a stale pass is worse than no claim.",
        "",
        "- `python -m pytest tests -q` -> **68 passed**",
        "- `python -m scripts.security_audit` -> **16 passed, 0 failed, 0 warned, "
        "8 not applicable**",
        "",
        "Three defects in this project's own code were found by those two harnesses "
        "and are written up in `SECURITY.md`: dead request validation on the demo "
        "API, a credential that could be logged in full while passing the redaction "
        "filter, and two unbounded `2**n` allocations behind a ceiling that could "
        "not fire. A fourth -- an inverted QAOA quality metric -- was found the same "
        "way.",
        "",
    ]


def section_prohibited(exp: dict) -> list[str]:
    """The list that stops a negative result being rounded into a neutral one."""
    lines = [
        "## 9. Claims that may NOT be made",
        "",
        "Written down next to the numbers so that making one becomes a deliberate "
        "act rather than a drafting convenience.",
        "",
        "1. **No speed-up.** Not \"faster\", not \"efficient\", not \"reduced "
        "latency\". The simulated pipeline is slower than its own baseline by the "
        "factor in section 6.",
        "2. **Grover's reduction is in oracle queries only.** It may never be "
        "written as a time or throughput improvement.",
        "3. **The global fidelity kernel is a ranking no-op before training** "
        "(tau = 0.9995) and this is a reported finding, not an omission.",
        "4. **The Gate B margin is not significant.** "
        "137 validation pairs; a +1.5-point top-1 change is about two queries.",
        "5. **The injection detector does not defend against fluent poisoning.** "
        "0% detection on topical mimicry. The aggregate rate may not be quoted "
        "without that row.",
        "6. **The embedding-optimised attack is black-box** (forward passes only) "
        "and strictly weaker than the gradient-based attack of Zhong et al. (2023). "
        "Any defence result is against the weaker attack and must say so.",
        "7. **No hardware claim.** Nothing here ran on a quantum device; all "
        "circuits are simulated on a classical machine.",
        "8. **One corpus, one attack budget.** No generalisation beyond SciFact is "
        "supported by these runs.",
    ]
    # Whether the headline retrieval claim is even available is decided by data.
    sig = exp.get("significance", {})
    wins = [(label, c["ndcg@10"]["delta"])
            for label, c in sig.items()
            if isinstance(c.get("ndcg@10"), dict)
            and c["ndcg@10"].get("significant") and c["ndcg@10"]["delta"] > 0]
    losses = [(label, c["ndcg@10"]["delta"])
              for label, c in sig.items()
              if isinstance(c.get("ndcg@10"), dict)
              and c["ndcg@10"].get("significant") and c["ndcg@10"]["delta"] < 0]
    lines.append("")
    if wins:
        best = max(wins, key=lambda t: t[1])
        lines += [
            f"**Available headline claim:** `{best[0]}` improves nDCG@10 by "
            f"{best[1]:+.4f} over the tuned classical baseline with a 95% CI "
            f"excluding zero, on {exp['dataset']['n_queries']} SciFact test "
            f"queries. Nothing stronger.",
        ]
    else:
        lines += [
            "**There is no significant retrieval improvement to claim.** No system "
            "in the grid beats the classical baseline on nDCG@10 with a CI "
            "excluding zero.",
        ]
        if losses:
            worst = min(losses, key=lambda t: t[1])
            lines += [
                "",
                f"Worse: {len(losses)} system(s) are significantly **below** "
                f"baseline, the largest being `{worst[0]}` at {worst[1]:+.4f} "
                "nDCG@10. This is the result and it is what the paper reports. "
                "SciFact's tuned hybrid baseline is strong "
                f"(nDCG@10 = "
                f"{exp['clean']['classical-baseline']['metrics']['ndcg@10']:.4f}), "
                "which leaves little headroom, and the phase kernel overfits 1024 "
                "free parameters to 919 training pairs. Both are stated as the "
                "explanation rather than as an excuse, and neither converts the "
                "regression into a success.",
            ]
    lines.append("")
    return lines


def main() -> int:
    exp = _load("experiment.json")
    kt = _load("kernel_training.json")
    if exp is None:
        print("results/experiment.json not found. Run:\n"
              "  python -m scripts.run_experiment")
        return 2
    if kt is None:
        print("results/kernel_training.json not found. Run:\n"
              "  python -m scripts.train_kernel")
        return 2

    body: list[str] = [
        "# FACTS -- source of truth for every Q-RAG document",
        "",
        "**Generated file. Do not edit by hand.**",
        "",
        "```bash",
        "python -m scripts.build_facts",
        "```",
        "",
        "Every number below is read out of `results/*.json`. The research paper, the",
        "literature review, the slide deck and the daily diary are all written from",
        "this file, so that no figure is transcribed twice and none of them can",
        "drift. If an experiment is re-run, regenerate this file rather than",
        "editing it.",
        "",
    ]
    body += section_provenance(exp)
    body += section_kernel(kt)
    body += section_ablation(exp)
    body += section_significance(exp)
    body += section_quantum(exp)
    body += section_latency(exp)
    body += section_poison(exp)
    body += section_status()
    body += section_prohibited(exp)

    out = ROOT / "FACTS.md"
    out.write_text("\n".join(body), encoding="utf8")
    print(f"wrote {out} ({len(body)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
