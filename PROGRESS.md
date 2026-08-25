# PROGRESS

Resume file. If the session died, this plus the git log is enough to pick up.

Updated after each deliverable is committed and pushed. Last update: 26 Aug 2026,
early morning, ahead of the 07:25 cutoff.

## Hard constraints in force

- **07:25 cutoff.** Laptop leaves at 07:30. Stop starting new work at 07:25, commit
  and push everything, print a status block.
- **No new timed experiment run after 06:45.** `perf_counter` keeps counting through
  a suspend, so any `latency_ms` or `run_seconds` from a run that straddles the
  shutdown is corrupt. `results/experiment.json` is already complete — do not re-run
  it. `scripts/verify.py` is *not* timed and is safe to run at any point.
- **Every number in the paper, deck and diary is read from `results/*.json`,** never
  typed. Enforced by `docs/facts.py`: documents contain `{{token}}` placeholders and
  an unknown token fails the build.
- **No fabricated citations.** Real, verifiable references only.
- **Negative results are reported as negative.** No weight tuning to improve a table.

## Deliverable status

| # | Deliverable | Path | State |
|---|---|---|---|
| 1 | Source-of-truth facts file | `FACTS.md`, `scripts/build_facts.py`, `docs/facts.py` | **done, pushed** |
| 2 | README + SECURITY | `README.md`, `SECURITY.md` | next |
| 3 | Daily diary DOCX | `docs/daily_diary.md` -> `build/Q-RAG_Daily_Diary.docx` | not started |
| 4 | 10-slide PPTX | `docs/slides.py` -> `build/Q-RAG_Slides.pptx` | not started |
| 5 | Research paper DOCX | `docs/research_paper.md` -> `build/Q-RAG_Research_Paper.docx` | not started |
| 6 | Literature review DOCX | `docs/literature_review.md` -> `build/Q-RAG_Literature_Review.docx` | prose complete, renders at ~24 est. pages; commit last per the agreed order |

## What deliverable 1 actually contains

`FACTS.md` is generated, not written. Three mechanisms sit behind it:

- `scripts/build_facts.py` reads `results/*.json` and emits all 9 sections of
  `FACTS.md`, including the list of claims the project may not make.
- `docs/facts.py` resolves `{{token}}` placeholders in document sources at render
  time. `python -m docs.facts` lists every available token and its current value.
- `scripts/verify.py` runs the test suite and the security audit and records what
  they returned in `results/verification.json`, so the "68 tests pass, 16 audit
  checks pass" figures are read rather than remembered.

## Things found while building it (all fixed, none papered over)

1. **The security audit had silently regressed to 15 pass / 1 warn.** Check PROD-3
   flagged `results/experiment_poison.json` — the mid-run checkpoint — as a result
   file with no embedded config, so its numbers were untraceable. Fixed three ways:
   the checkpoint writer now embeds provenance; the stale file was deleted after
   verifying it was a strict subset of the final `experiment.json` with identical
   values; and `results/*_poison.json` is now gitignored. Back to 16 pass / 0 warn
   legitimately, not by suppressing the check.
2. **PROD-3 was testing for a literal `config` key rather than for traceability.**
   The verification artefacts have no retrieval config to hash. The check now accepts
   a config hash *or* a git commit, and still flags a file carrying neither — which
   is the case that caught the checkpoint.
3. **README claims "one cell of 35" significance comparisons. The real number is
   30** (6 systems x 5 metrics), which the `sig.total` token computes from the
   results file. Fix belongs to deliverable 2.
4. Provenance was duplicated across result writers; it is now one shared
   `qrag/provenance.py`.

## The headline results, so a resumed session does not have to re-derive them

Read from `results/experiment.json`. Full detail in `FACTS.md`.

- **Retrieval: null.** 30 comparison cells across 6 systems, paired bootstrap,
  2,000 resamples. One cell reaches p < 0.05 (`qrag[kernel]` recall@5, +0.0173,
  p = 0.047); with 30 comparisons that is noise and is not claimed as a finding.
  The kernel passed both pre-registered gates on 137 held-out pairs and did not
  transfer to 300 queries.
- **Grover: 4.64x fewer oracle queries, 8.98x simulation overhead.** Scores
  bit-identical to the baseline on every retrieval metric — it selects from an
  already-scored shortlist, so it contributes nothing to ranking quality.
- **QAOA: 0.9978 mean solution quality**, exact optimum on 78.7% of queries, at
  ~1,122 ms/query simulated. Most expensive stage by a factor of thirty.
- **Security: the one positive result, and it is narrow.** Adversarial context
  occupancy 0.9840 (`qrag[no-qaoa]`) -> 0.7960 (`qrag[full]`), a 0.1880 reduction
  attributable to the redundancy penalty via an ablation differing in that term
  alone. 0.7960 is still catastrophic: all 50 targeted queries hit, no clean
  context, detector catches 100% of instruction-injection and 0% of the other three
  families.
- **Full pipeline is 30.7x slower than its own classical baseline.** No speed-up is
  claimed anywhere.

## Known gaps, stated rather than hidden

- Two placeholders in the literature review front matter I cannot fill and must not
  invent: `[ADD REMAINING GROUP 165 MEMBERS]` and
  `[ADD SUPERVISOR NAME AND DESIGNATION]`.
- Page counts printed by `python -m docs.build` are estimates. python-docx writes
  the file but does not lay it out, so the true count depends on the renderer that
  opens it. Open the DOCX in Word to confirm the 20-page requirement.
- `qrag/generate.py` (end-to-end answer generation and answer-relevancy judging) was
  scoped but is not built. Nothing in the five deliverables depends on it.
