---
title: Quantum-Enhanced Retrieval-Augmented Generation
subtitle: Daily Work Diary, Weeks 1-6
degree: Semester-7 Major Project, Group 165, B.Tech Computer Science and Engineering
authors: Prabhav Goel; [ADD REMAINING GROUP 165 MEMBERS]
supervisor: [ADD SUPERVISOR NAME AND DESIGNATION]
department: Department of Computer Science and Engineering
institution: Amity School of Engineering and Technology
date: August 2026
footer: Q-RAG Daily Diary
---

# Daily Work Diary

Weeks 1 to 6, Monday 20 July 2026 to Wednesday 26 August 2026. Week 6 is in
progress at the time of writing; entries stop at Wednesday 26 August.

Every measured figure quoted in this diary is read directly from
`results/experiment.json` and `results/kernel_training.json` when the document is
built, not typed in. Configuration hash `{{config_hash}}`, seed `{{seed}}`.

[[TOC]]

[[PAGEBREAK]]

## Week 1 — Monday 20 to Sunday 26 July

Environment, corpus and a working classical baseline. The week's principle was that
nothing quantum gets written until there is a classical system good enough to be
embarrassed by.

| Date | Day | Work done |
|---|---|---|
| 20 Jul | Mon | Repository initialised. Python 3.13 environment, dependency list pinned. Read the approved synopsis and listed the three quantum stages it commits to. |
| 21 Jul | Tue | BEIR SciFact loader written. Corpus verified at {{n_docs}} abstracts, {{n_queries}} test queries, {{n_judgments}} relevance judgments. |
| 22 Jul | Wed | Embedding backend abstracted over Ollama, sentence-transformers and a hashing stub. Content-addressed cache so a re-run costs nothing. |
| 23 Jul | Thu | Corpus and query embeddings built with `bge-m3`, dimension {{embed_dim}}. Cache came to 25 MB, gitignored. |
| 24 Jul | Fri | BM25 index and a dense (faiss) index. Sanity-checked both by eye on ten queries. |
| 25 Jul | Sat | Metrics module: recall, precision, MRR, nDCG at several cut-offs. Per-query scores retained, not just means. |
| 26 Jul | Sun | Light day. Read the BEIR paper properly and noted that SciFact has a strong lexical baseline, which makes it a hard place to show a gain. |

## Week 2 — Monday 27 July to Sunday 2 August

Fusion, the configuration system, and the discovery that reshaped the project.

| Date | Day | Work done |
|---|---|---|
| 27 Jul | Mon | Per-query min-max normalisation and weighted fusion of BM25 and cosine. First hybrid baseline running end to end. |
| 28 Jul | Tue | Dataclass config, hashed into every result file, so a table can be traced back to the run that produced it. All seeds fixed at `{{seed}}`. |
| 29 Jul | Wed | Wrote the numpy statevector simulator. Amplitude encoding, normalisation, measurement probabilities. |
| 30 Jul | Thu | Cross-checked every fast path against qiskit-aer. Two disagreements found and both were my simulator's fault, not aer's. |
| 31 Jul | Fri | **Derived the global fidelity kernel and found it is a ranking no-op.** At theta = 0 it equals cos-squared, which is rank-identical to cosine. A quantum stage that cannot reorder anything makes the whole comparison vacuous. |
| 1 Aug | Sat | Sat with the no-op result rather than coding around it. Wrote it up as the project's central risk instead of hiding it. |
| 2 Aug | Sun | Rest day. |

## Week 3 — Monday 3 to Sunday 9 August

Answering the no-op: a kernel that breaks rank-equivalence structurally, and two
gates defined before any training so the answer could not be judged after the fact.

| Date | Day | Work done |
|---|---|---|
| 3 Aug | Mon | Designed the block (projected) fidelity kernel. At theta = 0 it equals a sum of squared block similarities, which by Cauchy-Schwarz is not monotone in cos-squared — so it can reorder where the global kernel cannot. |
| 4 Aug | Tue | **Defined both gates before training.** Gate A: Kendall tau against cosine must fall below {{gate_a.threshold}}. Gate B: fused MRR must beat cosine MRR on held-out pairs. |
| 5 Aug | Wed | InfoNCE training loop with hand-derived analytic gradients. No autodiff, so the gradient had to be checked numerically. |
| 6 Aug | Thu | Hard-negative mining, pool of 64 per query. Built {{n_train_pairs}} training pairs from the train split. |
| 7 Aug | Fri | Trained both kernels. Block kernel: {{n_blocks}} blocks of {{block_size}} dimensions. Early stopping on validation loss. |
| 8 Aug | Sat | **Both gates pass.** Gate A: tau {{gate_a.tau}}, comfortably under {{gate_a.threshold}}. Gate B: MRR {{gate_b.mrr_before}} to {{gate_b.mrr_after}} on {{n_val_pairs}} held-out pairs. |
| 9 Aug | Sun | Recorded the honest caveats beside the pass: the Gate B margin is about two queries, and {{n_blocks}} times {{block_size}} free phases against {{n_train_pairs}} pairs will overfit. |

## Week 4 — Monday 10 to Sunday 16 August

The two quantum subroutines, and a bug that had been flattering the results.

| Date | Day | Work done |
|---|---|---|
| 10 Aug | Mon | Grover amplitude amplification with a threshold oracle over a pre-scored shortlist. {{grover.qubits}} qubits over {{grover.candidates}} candidates. |
| 11 Aug | Tue | Decided to report Grover as **oracle queries**, never as wall-clock. {{grover.oracle_queries}} query against {{grover.classical_queries}} expected for a classical scan is a {{grover.reduction}}x reduction; simulating it costs {{grover_arm.overhead}}x more time. Both go in adjacent columns. |
| 12 Aug | Wed | Context selection written as a QUBO: relevance reward, pairwise redundancy penalty, cardinality constraint. |
| 13 Aug | Thu | QAOA over that QUBO, {{qaoa.qubits}} qubits, p = {{qaoa.layers}} layers, COBYLA with restarts. Exact brute force computed alongside at small n so quality is measured against a true optimum. |
| 14 Aug | Fri | **Found the quality metric was inverted.** It reported a ratio above 1.0, which read literally claims QAOA beat the exact optimum. Replaced with an affine-invariant quality bounded by 1.0 against the same feasible set. |
| 15 Aug | Sat | Re-measured after the fix: {{qaoa.quality}} mean quality, exact optimum on {{qaoa.exact_rate|pct}} of queries, worst case {{qaoa.quality_worst}}. Simulation cost {{qaoa.ms}} ms/query. |
| 16 Aug | Sun | Sub-query interference reranking with decaying weights. Working, small effect. |

## Week 5 — Monday 17 to Sunday 23 August

Adversarial arm, security hardening, and the experiment of record.

| Date | Day | Work done |
|---|---|---|
| 17 Aug | Mon | Four attack families implemented: topical mimicry, lexical gaming, embedding-optimised, instruction injection. Relevance judgments deliberately left unmodified so retrieval metrics stay comparable. |
| 18 Aug | Tue | Poisoning harness: {{poison.n_injected}} passages against {{poison.n_targets}} target queries. Recorded that the embedding-optimised attack is black-box and therefore weaker than a gradient-based one. |
| 19 Aug | Wed | Security work against the pre-deploy checklist: input validation, resource ceilings on the `2**n` allocation paths, log redaction with salted query fingerprints, restricted CORS, rate limiting. |
| 20 Aug | Thu | Wrote `security_audit.py`. It found three real defects in my own code: dead request validation, a credential that could be logged in full while passing the redaction filter, and two unbounded allocations behind a ceiling that could not fire. |
| 21 Aug | Fri | Test suite to {{tests.passed}} tests, with a standing rule that no test asserts a research result — a test like `assert mrr > 0.7` turns a measurement into a requirement. |
| 22 Aug | Sat | Added `qrag[no-qaoa]`, the ablation control that makes the security hypothesis falsifiable. Without it, any drop in adversarial occupancy could be credited to the kernel instead. |
| 23 Aug | Sun | Ran the experiment of record: {{n_systems}} configurations over all {{n_queries}} queries, paired bootstrap with 2,000 resamples, plus the poisoned arm. |

## Week 6 — Monday 24 to Wednesday 26 August (in progress)

Reading the results honestly and writing them up. This week is not finished; entries
end on Wednesday 26 August.

| Date | Day | Work done |
|---|---|---|
| 24 Aug | Mon | **Read the result: retrieval is null.** {{sig.count}} of {{sig.total}} significance cells crosses p < 0.05, and with {{sig.total}} comparisons that is noise. The kernel passed both gates on held-out pairs and did not transfer to {{n_queries}} queries. Reported as negative rather than tuned. |
| 25 Aug | Tue | Security result, which is the one positive: adversarial context occupancy {{poison.occ_noqaoa}} without QAOA to {{poison.occ_full}} with it, a reduction of {{poison.qaoa_effect}} attributable to the redundancy penalty alone. Counterweight recorded in the same paragraph — {{poison.occ_full}} is still catastrophic, with every target query hit and a clean-context rate of {{poison.clean_rate_full}}. |
| 26 Aug | Wed | Built the generated facts file so the five deliverables cannot quote different numbers, and a substitution layer that fails the build on an unknown figure. It immediately caught four stale figures in prose I had already written, and an audit check that had silently regressed to {{audit.passed}} passing only after a stale checkpoint was removed. Literature review, paper, deck and this diary in progress. |

[[PAGEBREAK]]

## What the six weeks produced

Stated plainly, including the parts that did not work.

**The headline retrieval result is negative.** Across {{n_queries}} queries and
{{n_systems}} configurations, no quantum arm beats the tuned classical baseline on
nDCG@10 with a confidence interval excluding zero. The baseline reaches nDCG@10
{{base.ndcg@10}} and the full pipeline {{full.ndcg@10}}. The kernel cleared both
pre-registered gates on {{n_val_pairs}} held-out pairs and then failed to transfer.

**No speed-up is claimed anywhere.** The full pipeline is {{slowdown}}x slower than
its own classical baseline, which is the expected cost of simulating quantum circuits
on a classical machine. Grover's {{grover.reduction}}x reduction is in oracle
queries, a hardware-independent count, and is never presented as a timing result.

**The security result is the one positive finding and it is narrow.** The redundancy
penalty reduced adversarial context occupancy by {{poison.qaoa_effect}}, isolated by
an ablation differing in that term alone. It did not come close to defeating the
attack, and the pattern detector catches
{{detector.instruction-injection|pct}} of instruction injection but
{{detector.topical-mimicry|pct}} of fluent topical mimicry.

**Verification, re-run on the last day rather than remembered:**
{{tests.passed}} tests pass, and the security audit reports {{audit.passed}} passed,
{{audit.failed}} failed, {{audit.warned}} warned, {{audit.na}} not applicable.
