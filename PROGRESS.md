# PROGRESS

Resume file. If the session died, this plus the git log is enough to pick up.

Updated after each deliverable is committed and pushed. Last update: 15 Oct 2026,
for the licensing decision below.

## Software follow-up in progress — 14 Oct 2026

- Streamlit dark workbench is pushed: live Classical / Kernel / Full modes,
  grounded answer and independent judge, recorded evaluation and security views,
  and ranked-passage CSV export.
- SRS follow-up implements session-local TXT/PDF ingestion with overlapping
  chunks, in-memory FAISS/BM25 indexing, Grover bypass for tiny corpora, and a
  numerical-failure QAOA fallback. The evaluation runner records per-query
  latency, ranks, and scores in a structured CSV. The full experiment was rerun
  and the result files and dependent documents were refreshed on 14 Oct.
- COBYLA routinely uses its iteration budget while producing finite feasible
  selections. The SRS wording on "failure to converge" needs revision or a
  different stopping policy; automatically discarding those selections would
  change the research question and its reported results.
- The project uses PolyForm Noncommercial License 1.0.0. The SRS and paper were
  corrected on 15 Oct to match the repository license; no relicensing occurred.

## Historical run constraints

- The 26 Aug 07:25 laptop cutoff no longer applies. The experiment was rerun
  uninterrupted on 13 Oct to add the greedy-MMR poisoned-corpus control.
- **Every number in the paper, deck and diary is read from `results/*.json`,** never
  typed. Enforced by `docs/facts.py`: documents contain `{{token}}` placeholders and
  an unknown token fails the build.
- **No fabricated citations.** Real, verifiable references only.
- **Negative results are reported as negative.** No weight tuning to improve a table.

## Deliverable status

| # | Deliverable | Path | State |
|---|---|---|---|
| 1 | Source-of-truth facts file | `FACTS.md`, `scripts/build_facts.py`, `docs/facts.py` | **done, pushed** |
| 2 | README + SECURITY | `README.md`, `SECURITY.md` | **done, pushed** |
| 3 | Daily diary DOCX | `docs/daily_diary.md` -> `deliverables/Q-RAG_Daily_Diary.docx` | **done** (dated October postscript, 9 pages in Word on 14 Oct) |
| 4 | 10-slide PPTX | `docs/slides.py` -> `deliverables/Q-RAG_Slides.pptx` | **done** (10 slides, every figure via `docs.facts`) |
| 5 | Research paper DOCX | `docs/research_paper.md` -> `deliverables/Q-RAG_Research_Paper.docx` | **done** (27 pages in Word on 14 Oct; measurements substituted from facts) |
| 6 | Literature review DOCX | `docs/literature_review.md` -> `deliverables/Q-RAG_Literature_Review.docx` | **done** (39 pages in Word on 14 Oct; exceeds the 20-page minimum) |

## What deliverable 1 actually contains

`FACTS.md` is generated, not written. Three mechanisms sit behind it:

- `scripts/build_facts.py` reads `results/*.json` and emits all 9 sections of
  `FACTS.md`, including the list of claims the project may not make.
- `docs/facts.py` resolves `{{token}}` placeholders in document sources at render
  time. `python -m docs.facts` lists every available token and its current value.
- `scripts/verify.py` runs the test suite and the security audit and records what
  they returned in `results/verification.json`, so the "80 tests pass, 16 audit
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
3. **README claimed "one cell of 35" significance comparisons; the real number is
   30** (6 systems x 5 metrics), which the `sig.total` token computes from the
   results file. Fixed in deliverable 2, along with two more stale figures the
   sweep caught: README said the pipeline was 31x slower (30.7x) and attributed
   the whole pipeline's 1,191 ms/query to the QAOA stage alone (1,122 ms/query);
   SECURITY.md still carried the long-superseded "~90x slower" and a stale
   tracked-file count.
4. Provenance was duplicated across result writers; it is now one shared
   `qrag/provenance.py`.
5. **A draft slide claimed QAOA cost "thirty times the rest of the pipeline".** It
   does not — that conflated the pipeline's 30.7x slowdown against the classical
   baseline with the QAOA stage's share of pipeline latency. Against the other four
   stages it is 16.3x; against the classical baseline's whole per-query cost, 28.9x.
   All three are now tokens computed in `docs/facts.py`, so the next document to
   quote the figure cannot re-derive it wrongly.
6. **A draft of the paper's Table 5.3 mixed two arms in one column.** Candidate
   count, marked count, oracle queries and success probability were taken from
   `qrag[full]`'s Grover block while the overhead and wall clock came from the
   `qrag[grover]` arm — 8.98x and 0.074 ms/query instead of the full pipeline's
   8.03x and 0.068 ms/query. Both tokens resolve, so the facts layer could not catch
   this one; it needed reading the table against the results file. The table is now
   one arm throughout, with the Grover-only arm's overhead on its own labelled row.
7. **The QAOA solution-quality figure is conditional, and the paper now says so.**
   The cardinality constraint is a soft penalty, so only 0.40 of the readout
   probability mass is feasible and `rerank_qaoa` post-selects before decoding
   (`qrag/qaoa.py:255-262`). Quoting 0.9978 without the 0.40 overstates the routine:
   on hardware it implies ~2.5x the shots for the same answer. Both now appear in
   the same table.
8. **The missing greedy-MMR control has now been run.** `qrag[full-mmr]` reaches
   0.9800 adversarial context occupancy, versus 0.7960 for `qrag[full]`; the cheap
   classical heuristic does not reproduce QAOA's reduction on this attack.

## The headline results, so a resumed session does not have to re-derive them

Read from `results/experiment.json`. Full detail in `FACTS.md`.

- **Retrieval: null.** 30 comparison cells across 6 systems, paired bootstrap,
  2,000 resamples. One cell reaches p < 0.05 (`qrag[kernel]` recall@5, +0.0173,
  p = 0.047); with 30 comparisons that is noise and is not claimed as a finding.
  The kernel passed both pre-registered gates on 137 held-out pairs and did not
  transfer to 300 queries.
- **Grover: 4.64x fewer oracle queries, 8.14x simulation overhead.** Scores
  bit-identical to the baseline on every retrieval metric — it selects from an
  already-scored shortlist, so it contributes nothing to ranking quality.
- **QAOA: 0.9978 mean solution quality**, exact optimum on 78.7% of queries, at
  ~913 ms/query simulated. That is 93.3% of the pipeline's per-query latency,
  13.9x the other four stages combined and 32.9x the classical baseline's entire
  per-query cost — three different ratios, all now `qaoa.share`, `qaoa.vs_rest` and
  `qaoa.vs_base` tokens rather than an approximate "factor of thirty".
- **Security: the one positive result, and it is narrow.** Adversarial context
  occupancy 0.9840 (`qrag[no-qaoa]`) -> 0.7960 (`qrag[full]`), a 0.1880 reduction
  attributable to the redundancy penalty via an ablation differing in that term
  alone. 0.7960 is still catastrophic: all 50 targeted queries hit, no clean
  context, detector catches 100% of instruction-injection and 0% of the other three
  families.
- **Full pipeline is 35.2x slower than its own classical baseline.** No speed-up is
  claimed anywhere.
- **QAOA quality is post-selected.** 0.9978 is the quality of the best *feasible*
  decoded selection; feasible readout mass is only 0.40. On hardware that is ~2.5x
  the shots. Never quote the quality without the feasible probability.

## What is next

The software, experiment evidence, and document sources are committed and pushed.
GitHub CI passed on commit `98c9c43` on 14 Oct. Extended CodeQL also scanned
that commit and reports **0 open, 5 closed** alerts. The live cached-corpus
`/search` and `/answer` paths returned 200, including generation and an
independent judge response. The 16 Oct submission checks below still need
final confirmation.

Before submitting, check the final document render:

1. **Verify the front matter with the group.** The paper, literature review and
   diary now use the names and project-guide role in the supplied SRS. The SRS
   does not state Dr Rani's formal academic rank, so only "Project Guide" is used.
2. **Review the visual layout in Word.** Automated Word
   pagination returned 39 literature review, 27 paper and 9 diary pages on
   14 Oct; TOC fields were updated and PDFs exported. A human visual pass is
   still needed. The 20-page literature review minimum is met on this renderer.
3. **Have a teammate rerun the project independently.** This has not been
   evidenced; do not claim it as complete.
4. **Check licensing consistency in the final exports.** The user selected the
   existing PolyForm Noncommercial License 1.0.0; the SRS and paper now match.
5. **Keep a second copy of the local submission ZIP.**
   `build/Q-RAG_Submission_2026-10-14.zip` contains tracked source and reports,
   cached SciFact data/embeddings, and three PDF previews. It excludes Ollama
   model weights, which must be installed separately.

Rebuild everything at any time with:

    python -m docs.build all && python -m docs.slides

That is safe to run — it reads `results/*.json` and never re-runs the experiment.

## Known gaps, stated rather than hidden

- The SRS names Dr Rani as project guide but gives no academic rank. Confirm any
  required formal title before submission.
- Page counts depend on the renderer. Word reported 39/27/9 pages for the
  literature review/paper/diary on this machine; visual layout and TOC updates
  still need a human final pass before submission.
- `qrag/generate.py` and `/answer` provide grounded Ollama generation plus an
  independent-model relevance/grounding judgement. The module-level API now loads
  the cached corpus embeddings and trained kernel automatically at startup.
