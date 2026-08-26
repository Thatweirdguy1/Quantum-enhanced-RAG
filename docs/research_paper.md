---
title: Quantum-Enhanced Retrieval-Augmented Generation
subtitle: A Measured Evaluation of Three Simulated Quantum Stages
degree: Semester-7 Major Project, Group 165, B.Tech Computer Science and Engineering
authors: Prabhav Goel; [ADD REMAINING GROUP 165 MEMBERS]
supervisor: [ADD SUPERVISOR NAME AND DESIGNATION]
department: Department of Computer Science and Engineering
institution: Amity School of Engineering and Technology
date: August 2026
footer: Q-RAG Research Paper
---

# Abstract

We build and evaluate Q-RAG, a retrieval-augmented generation pipeline in which
three stages are replaced by classically simulated quantum routines: a trained
phase-sensitive fidelity kernel in the retrieval score, Grover amplitude
amplification over a pre-scored candidate shortlist, and a Quantum Approximate
Optimization Algorithm (QAOA) solver for context passage selection. The evaluation
is designed so that each stage can be shown to contribute nothing, and two of the
three are.

The paper's first contribution is negative and structural. A fidelity kernel over
amplitude-encoded unit vectors reduces, at zero phase, to the square of the cosine
similarity, and squaring is monotone, so such a kernel induces exactly the ranking
it was introduced to improve. We measure this: on our corpus the untrained global
fidelity kernel agrees with the cosine ranking at Kendall
tau = {{gate_a.tau_global_before}}. We then give a kernel that escapes the
equivalence structurally rather than by hoping training will break it — a block
(projected) fidelity kernel whose zero-phase limit is a sum of squared block
similarities, which by the Cauchy-Schwarz inequality is not a monotone function of
cosine — and we verify the escape with two criteria fixed before any training.

The paper's second contribution is a null result reported as one. Across
{{n_systems}} configurations and all {{n_queries}} queries of the BEIR SciFact test
split, no quantum configuration improves on a tuned classical hybrid baseline with a
bootstrap confidence interval excluding zero. Exactly {{sig.count}} of
{{sig.total}} system-metric comparisons reaches p < 0.05, which is what
{{sig.total}} comparisons produce by chance. The trained kernel passed both
pre-registered criteria on {{n_val_pairs}} held-out pairs and did not transfer. The
full pipeline is {{slowdown}} times slower than its own baseline, and we claim no
speed-up anywhere: Grover's {{grover.reduction|2f}}-fold advantage is reported in
oracle queries, a hardware-independent count, alongside the
{{grover.overhead|2f}}-fold cost of simulating it.

The one positive finding is narrow and lies in security. On a corpus poisoned with
{{poison.n_injected}} adversarial passages across {{poison.n_families}} attack
families, QAOA's pairwise redundancy penalty reduces adversarial context occupancy
from {{poison.occ_noqaoa}} to {{poison.occ_full}}, a reduction of
{{poison.qaoa_effect}} isolated by an ablation whose two arms differ in that term
alone. We report it together with the fact that {{poison.occ_full}} occupancy is
still a total compromise: all {{poison.n_targets}} targeted queries are hit and the
clean-context rate is {{poison.clean_rate_full}}.

---

[[TOC]]

[[PAGEBREAK]]

# 1. Introduction

## 1.1 The claim under test

Retrieval-augmented generation (RAG) grounds a language model in a document
collection by retrieving passages relevant to a query and conditioning generation on
them (Lewis et al., 2020). Its two ranking components — a lexical scorer such as
BM25 (Robertson and Zaragoza, 2009) and a dense bi-encoder (Karpukhin et al., 2020)
— reduce to computing similarities in a vector space, and a substantial literature
proposes that quantum or quantum-inspired representations improve on the geometry
those similarities assume (van Rijsbergen, 2004; Piwowarski et al., 2010; Uprety et
al., 2020).

This paper tests a specific version of that proposal on a specific corpus, and it is
organised around a single methodological commitment: every stage is instrumented so
that "this quantum component does nothing" is a result the experiment can return.
That commitment is not rhetorical. It determined the choice of baseline, the decision
to fix the kernel acceptance criteria before training, the decision to report
Grover's advantage in oracle queries rather than seconds, and the inclusion of an
ablation arm whose only purpose is to make the security hypothesis falsifiable.

## 1.2 The failure mode this design is built around

The reason for that commitment is a recurring problem in the applied quantum
retrieval literature. A scoring function is introduced as quantum; it is shown to be
expressible as a state overlap or fidelity; and on inspection the overlap turns out
to be a monotone transformation of the classical similarity it was meant to replace.
A monotone transformation cannot reorder a ranked list. Where this holds, the quantum
stage is a no-op with respect to the retrieval task, and any measured difference in
retrieval quality originated elsewhere in the system.

Concretely, encode a unit-norm query embedding *q* and document embedding *d* as
amplitude vectors and take the fidelity of the two states. With real, non-negative
amplitudes and no phase, the fidelity is

    K(q, d) = |Σ_i q_i d_i|² = cos²(q, d)

and since cos(q, d) is non-negative on unit vectors, squaring preserves order. Every
document keeps its rank. Section 3.3 gives the measured version of this statement and
Section 3.4 gives a construction that escapes it.

## 1.3 Contributions

1. **A structural escape from rank-equivalence, verified rather than asserted.** A
   block fidelity kernel whose zero-phase limit is not monotone in cosine, with
   Kendall tau against the cosine ranking measured both after training
   ({{gate_a.tau}}) and in a training-free control at zero phase
   ({{gate_a.tau_theta0}}), establishing that the construction and not the fitted
   phases is what breaks the equivalence.
2. **A null retrieval result at full corpus scale, reported as null.** {{sig.count}}
   of {{sig.total}} comparisons significant at p < 0.05 under a paired bootstrap over
   all {{n_queries}} queries, against a baseline tuned to be hard rather than a
   strawman.
3. **A reporting discipline for simulated quantum subroutines**, under which Grover's
   query complexity advantage and its simulation cost are printed adjacently and
   never substituted for one another.
4. **An adversarial evaluation with an ablation control**, yielding the paper's one
   positive result and bounding how much that result is worth.

## 1.4 What this paper does not claim

Stated at the outset rather than buried in a limitations section. No speed-up of any
kind is claimed; all circuits are classically simulated and the pipeline is
{{slowdown}} times slower than its own classical baseline. No retrieval improvement
is claimed. No claim is made about behaviour on quantum hardware, which we did not
use. The security result is a single measurement on one corpus at one attack budget
and is not a defence.

# 2. Related work

A full critical survey accompanies this paper as a separate literature review; this
section states only the positions the design depends on.

**Rank-equivalence and quantum kernels.** That a quantum model's decision function
can be written as a kernel method is now standard (Schuld, 2021; Schuld and Killoran,
2019), and fidelity kernels of the form |⟨φ(x)|φ(x')⟩|² are the canonical
construction (Havlicek et al., 2019). The conditions under which such a kernel can
offer an advantage are restrictive: separations are known for contrived,
cryptographically structured data (Liu, Arunachalam and Temme, 2021), while for
data-driven problems classical methods with access to the same data are often
competitive (Huang et al., 2021), and several proposed quantum advantages in
recommendation and linear algebra were subsequently dequantised (Tang, 2019; see also
Aaronson, 2015, on the fine print of such claims). The quantum language model and
complex-valued matching literature (Sordoni, Nie and Bengio, 2013; Zhang et al.,
2018; Li, Wang and Melucci, 2019; Wang et al., 2019) is where phase is argued to
carry retrieval-relevant information; our construction takes that argument seriously
and tests the weakest link in it, namely whether the resulting score can reorder at
all.

**Search and optimisation subroutines.** Grover's algorithm gives a quadratic
reduction in oracle queries for unstructured search (Grover, 1996), is optimal in
that model (Bennett et al., 1997), and extends to unknown numbers of marked items
(Boyer et al., 1998) and to amplitude amplification generally (Brassard et al.,
2002). The guarantee is in queries, not time. QAOA (Farhi, Goldstone and Gutmann,
2014) is applied to QUBO formulations (Lucas, 2014; Glover, Kochenberger and Du,
2019) of combinatorial problems; its performance at shallow depth is contested
(Zhou et al., 2020; Bravyi et al., 2020; Hastings, 2019), and NISQ-era variational
training faces barren plateaus (McClean et al., 2018; Cerezo et al., 2021; Bharti et
al., 2022; Preskill, 2018).

**RAG security.** Corpus poisoning against dense retrievers (Zhong et al., 2023) and
against RAG end-to-end (Zou et al., 2025) is effective at very low injection
budgets, and indirect prompt injection via retrieved content is a distinct threat
(Greshake et al., 2023; Perez and Ribeiro, 2022; OWASP, 2025). Defences are commonly
reported in aggregate; we report per-family detection because the aggregate conceals
which families a detector misses entirely.

# 3. Method

## 3.1 Pipeline

The system is a sequence of five stages, each independently switchable so that the
ablation grid in Section 4.2 is a matter of configuration rather than of separate
code paths.

| Stage | Function | Quantum? |
|---|---|---|
| Candidate generation | BM25 over the corpus, dense retrieval over `bge-m3` embeddings of dimension {{embed_dim}} | no |
| Fusion | per-query min-max normalisation, weighted sum of BM25, cosine and kernel scores | kernel term only |
| Interference rerank | sub-query decomposition, decaying weights, coherent recombination | simulated |
| Shortlist search | Grover amplitude amplification with a threshold oracle | simulated |
| Context selection | QAOA over a selection QUBO | simulated |

All quantum stages are classically simulated. A hand-written numpy statevector
implementation carries the experiment, and every routine is cross-checked against
qiskit-aer (Javadi-Abhari et al., 2024) in the test suite; the numpy path exists for
speed, not because the two disagree.

## 3.2 Fusion and the baseline

Scores from heterogeneous retrievers are not comparable in magnitude, so each score
vector is min-max normalised within a query before fusion, and the fused score is a
weighted sum. Rank-based fusion (Cormack, Clarke and Buettcher, 2009) would avoid the
normalisation question but discards score margins that the kernel term is meant to
alter, so we retain score-level fusion.

The baseline matters more than the treatment here. A classical baseline that is
merely BM25, or an untuned hybrid, makes any quantum arm look good. Our baseline is
the *same* fusion with the kernel weight redistributed onto the cosine term, so the
comparison isolates the kernel rather than the presence of a third scorer, and so
that the two systems have identical candidate sets and identical normalisation.

## 3.3 The rank-equivalence problem, measured

Section 1.2 gives the algebra. Its empirical form on our corpus is the following. We
constructed the global fidelity kernel with all phases at zero, scored the same
candidate sets with it and with plain cosine, and computed Kendall's tau (Kendall,
1938) between the two orderings per query:

> Mean Kendall tau between the untrained global fidelity ranking and the cosine
> ranking: **{{gate_a.tau_global_before}}**.

A value this close to unity means the two rankings are, for practical purposes, the
same list. Reporting it is the point: any subsequent retrieval difference attributed
to this kernel would have to have come from somewhere other than the kernel.

## 3.4 The block fidelity kernel

Partition the {{embed_dim}} embedding dimensions into {{n_blocks}} contiguous blocks
of {{block_size}} dimensions. Assign each dimension a trainable phase θ_i and each
block a non-negative weight w_g, and define

    K(q, d) = Σ_g w_g |Σ_{i ∈ g} q_i d_i e^{i θ_i}|²

Physically this is a projected fidelity: rather than one global overlap, the state is
measured block-wise and the outcomes are combined. The property that matters is its
zero-phase limit. Writing S_g = Σ_{i ∈ g} q_i d_i for the block-restricted inner
product and taking uniform weights, the kernel becomes Σ_g S_g², while cosine
similarity is Σ_g S_g. By the Cauchy-Schwarz inequality, Σ_g S_g² is *not* a monotone
function of (Σ_g S_g)²: two document pairs with the same total similarity but
different distributions of that similarity across blocks receive different kernel
scores, and their order can invert. The kernel can therefore reorder before any
training happens, which is what distinguishes it from the global construction of
Section 3.3.

## 3.5 Kernel training and the two acceptance criteria

Phases are trained with InfoNCE (van den Oord, Li and Vinyals, 2018) over
{{n_train_pairs}} query-positive pairs mined from the SciFact train split with
hard negatives (Xiong et al., 2021), using analytic gradients and Adam (Kingma and
Ba, 2015), with early stopping on {{n_val_pairs}} held-out validation pairs.

Two criteria were fixed **before** training, and are reported whether they pass or
fail:

| Criterion | Requirement | Rationale |
|---|---|---|
| A: does it reorder? | mean Kendall tau against cosine < {{gate_a.threshold}} | a kernel that cannot reorder cannot help or hurt, and a retrieval comparison built on it is vacuous |
| B: is the reordering an improvement? | fused MRR on held-out pairs must exceed cosine MRR | reordering alone is not progress; it must be reordering in the right direction |

Fixing them in advance is the only defence against the obvious failure mode, which is
choosing the criterion after seeing which one the trained kernel happens to satisfy.

## 3.6 Amplitude amplification over a scored shortlist

We do not use Grover to search the corpus. Unstructured quantum search over a
document collection requires the collection in superposition, which presumes a QRAM
whose cost is exactly the contested part of such proposals (Kerenidis and Prakash,
2017; Harrow, Hassidim and Lloyd, 2009; Aaronson, 2015). Instead we take an
already-scored shortlist of {{grover.candidates|0f}} candidates, define a threshold
oracle that marks those above a relevance cut-off, and run amplitude amplification
over the {{grover.qubits}}-qubit index register, with the iteration count set from
the estimated number of marked items (Boyer et al., 1998).

This is deliberately a modest claim. Because the shortlist is already ranked, the
retrieved set is unchanged by construction, and Section 5.3 confirms that the
Grover arm's retrieval metrics are bit-identical to the baseline's. What the stage
demonstrates is a query-complexity property on a real workload, and it is reported
in queries for that reason.

## 3.7 Context selection as a QUBO

Selecting *k* passages from *n* candidates for the generator's context is a trade-off
between relevance and redundancy, the classical treatment of which is maximal
marginal relevance (Carbonell and Goldstein, 1998). We write it as a QUBO over binary
selection variables x_i:

    minimise  −Σ_i r_i x_i  +  λ Σ_{i<j} s_ij x_i x_j  +  μ (Σ_i x_i − k)²

with r_i the fused relevance of candidate *i*, s_ij the cosine similarity between
candidates *i* and *j*, λ the redundancy penalty and μ the cardinality penalty. QAOA
is run over the corresponding Ising Hamiltonian (Lucas, 2014) at
p = {{qaoa.layers}} layers on {{qaoa.qubits}} qubits, with COBYLA (Powell, 1994) and
random restarts for the variational parameters.

Because *n* is small, we also compute the exact optimum by brute force over the
feasible set. Solution quality is then measured against a true optimum rather than
against another heuristic, and is defined affinely on the same exactly-*k* feasible
set so that it is bounded above by 1.0. An earlier version of this metric was a raw
objective ratio that could exceed 1.0 — that is, it could report QAOA beating the
exact optimum — which is a diagnostic of a broken metric rather than a good result,
and is recorded here because the corrected figures in Section 5.4 are lower.

## 3.8 Adversarial corpus construction

The poisoned arm injects {{poison.n_injected}} passages targeting
{{poison.n_targets}} test queries, across {{poison.n_families}} families:

| Family | Construction | Detector rate |
|---|---|---|
| topical-mimicry | fluent passage on the query's topic with a false claim | {{detector.topical-mimicry|pct}} |
| lexical-gaming | query terms repeated to exploit term-frequency saturation | {{detector.lexical-gaming|pct}} |
| embedding-optimised | black-box hill-climbing on cosine to the query embedding | {{detector.embedding-optimised|pct}} |
| instruction-injection | imperative text addressed to the generator | {{detector.instruction-injection|pct}} |

Two design decisions constrain what the arm can show. Relevance judgments are left
**unmodified**, so retrieval metrics on the poisoned corpus remain comparable with
the clean arm and a drop is interpretable. And the embedding-optimised attack is
black-box, using only similarity queries, which makes it strictly weaker than a
white-box gradient attack (Ebrahimi et al., 2018; Wallace et al., 2019) and means any
defence result here is a result against the weaker attack.

# 4. Experimental setup

## 4.1 Corpus and configuration

| Item | Value |
|---|---|
| Dataset | BEIR SciFact test split (Thakur et al., 2021; Wadden et al., 2020) |
| Documents | {{n_docs}} |
| Queries | {{n_queries}} (all of them; no sampling) |
| Relevance judgments | {{n_judgments}} |
| Embeddings | `bge-m3`, dimension {{embed_dim}} (Chen et al., 2024) |
| Dense index | faiss (Johnson, Douze and Jegou, 2021) |
| Configuration hash | `{{config_hash}}` |
| Seed | `{{seed}}` |
| Python / numpy | {{python}} / {{numpy}} |

SciFact is a deliberately unfavourable choice. It is a scientific claim-verification
collection with a strong lexical baseline, which is where a dense or exotic scorer
has least room to improve. Every result below is on all {{n_queries}} queries; the
configuration hash and seed reproduce the run.

## 4.2 Ablation grid

{{n_systems}} configurations: the classical baseline; the kernel alone; kernel plus
interference; Grover alone; QAOA alone; kernel plus QAOA; and the full pipeline. A
further configuration, `qrag[no-qaoa]`, is run on the poisoned corpus only. Its sole
purpose is to make the security hypothesis falsifiable: without an arm identical to
the full pipeline except for the QAOA stage, a change in adversarial occupancy could
be attributed to the kernel, to the interference rerank, or to the QAOA term, and
there would be no way to tell which.

## 4.3 Metrics and significance

Recall, precision, MRR and nDCG (Jarvelin and Kekalainen, 2002) at the cut-offs
recorded in the run configuration, {{config.eval.k_values}}, with per-query scores
retained rather than only means; the primary cut-off is
{{config.eval.primary_k}}. Differences against the baseline are tested with a paired
bootstrap over queries, {{config.eval.bootstrap_samples}} resamples (Efron and
Tibshirani, 1993; Sakai, 2006). A difference is called significant only when the 95%
percentile interval excludes zero. Latency is wall-clock per query on one machine,
reported as a mean, and is used only for cost accounting, never as evidence about
algorithmic complexity.

No test in the accompanying test suite asserts a research outcome. A test of the form
`assert ndcg > 0.7` converts a measurement into a requirement and creates an
incentive to adjust weights until it passes; the suite tests invariants — that the
simulator matches qiskit-aer, that fusion is scale-invariant, that the QUBO
objective is evaluated consistently — and lets the measurements be whatever they are.

# 5. Results

## 5.1 The kernel acceptance criteria

| Criterion | Requirement | Measured | Outcome |
|---|---|---|---|
| A: reorders | tau < {{gate_a.threshold}} | {{gate_a.tau}} | {{gate_a.pass}} |
| A control: untrained, zero phase | — | {{gate_a.tau_theta0}} | reorders without training |
| A control: global kernel | — | {{gate_a.tau_global_before}} | no-op, as predicted |
| B: fused MRR beats cosine | delta > 0 | {{gate_b.mrr_before}} to {{gate_b.mrr_after}} (delta {{gate_b.delta_mrr}}) | {{gate_b.pass}} |
| B: top-1 accuracy | — | {{gate_b.top1_before}} to {{gate_b.top1_after}} (delta {{gate_b.delta_top1}}) | — |

Both criteria pass. Kendall's tau is a similarity, so a *lower* value means more
reordering, and the zero-phase control is the informative row: at
{{gate_a.tau_theta0}} the untrained block kernel reorders slightly more than the
trained one at {{gate_a.tau}}, confirming that the escape from rank-equivalence is a
property of the block construction rather than of the fitted phases. Set against it,
the global kernel's {{gate_a.tau_global_before}} is the predicted no-op: with the
ceiling for "reorders at all" set at {{gate_a.threshold}}, the global kernel fails it,
and it fails it in the direction the algebra of Section 1.2 says it must.

Criterion B passes thinly and should not be read as more than it is. The margin is a
top-1 change of {{gate_b.delta_top1}} on {{n_val_pairs}} validation pairs, which is
about two queries, and {{n_blocks}} × {{block_size}} free phases against
{{n_train_pairs}} training pairs is a regime in which overfitting is expected.
Section 5.2 is where that expectation is tested.

## 5.2 Retrieval quality on the full corpus

| System | recall@10 | nDCG@10 | MRR@10 | ms/query |
|---|---|---|---|---|
| classical-baseline | {{clean.classical-baseline.metrics.recall@10}} | {{clean.classical-baseline.metrics.ndcg@10}} | {{clean.classical-baseline.metrics.mrr@10}} | {{clean.classical-baseline.metrics.latency_ms_mean}} |
| qrag[kernel] | {{clean.qrag[kernel].metrics.recall@10}} | {{clean.qrag[kernel].metrics.ndcg@10}} | {{clean.qrag[kernel].metrics.mrr@10}} | {{clean.qrag[kernel].metrics.latency_ms_mean}} |
| qrag[kernel+interf] | {{clean.qrag[kernel+interf].metrics.recall@10}} | {{clean.qrag[kernel+interf].metrics.ndcg@10}} | {{clean.qrag[kernel+interf].metrics.mrr@10}} | {{clean.qrag[kernel+interf].metrics.latency_ms_mean}} |
| qrag[grover] | {{clean.qrag[grover].metrics.recall@10}} | {{clean.qrag[grover].metrics.ndcg@10}} | {{clean.qrag[grover].metrics.mrr@10}} | {{clean.qrag[grover].metrics.latency_ms_mean}} |
| qrag[qaoa] | {{clean.qrag[qaoa].metrics.recall@10}} | {{clean.qrag[qaoa].metrics.ndcg@10}} | {{clean.qrag[qaoa].metrics.mrr@10}} | {{clean.qrag[qaoa].metrics.latency_ms_mean}} |
| qrag[kernel+qaoa] | {{clean.qrag[kernel+qaoa].metrics.recall@10}} | {{clean.qrag[kernel+qaoa].metrics.ndcg@10}} | {{clean.qrag[kernel+qaoa].metrics.mrr@10}} | {{clean.qrag[kernel+qaoa].metrics.latency_ms_mean}} |
| qrag[full] | {{clean.qrag[full].metrics.recall@10}} | {{clean.qrag[full].metrics.ndcg@10}} | {{clean.qrag[full].metrics.mrr@10}} | {{clean.qrag[full].metrics.latency_ms_mean}} |

[[CAPTION]] Table 5.1: Retrieval on all {{n_queries}} SciFact test queries. Config `{{config_hash}}`, seed `{{seed}}`.

Paired bootstrap against the baseline on nDCG@10:

| System | delta nDCG@10 | 95% CI | p | significant |
|---|---|---|---|---|
| qrag[kernel] | {{significance.qrag[kernel].ndcg@10.delta|signed}} | [{{significance.qrag[kernel].ndcg@10.ci95_low|4f}}, {{significance.qrag[kernel].ndcg@10.ci95_high|4f}}] | {{significance.qrag[kernel].ndcg@10.p_value|3f}} | {{significance.qrag[kernel].ndcg@10.significant}} |
| qrag[kernel+interf] | {{significance.qrag[kernel+interf].ndcg@10.delta|signed}} | [{{significance.qrag[kernel+interf].ndcg@10.ci95_low|4f}}, {{significance.qrag[kernel+interf].ndcg@10.ci95_high|4f}}] | {{significance.qrag[kernel+interf].ndcg@10.p_value|3f}} | {{significance.qrag[kernel+interf].ndcg@10.significant}} |
| qrag[grover] | {{significance.qrag[grover].ndcg@10.delta|signed}} | [{{significance.qrag[grover].ndcg@10.ci95_low|4f}}, {{significance.qrag[grover].ndcg@10.ci95_high|4f}}] | {{significance.qrag[grover].ndcg@10.p_value|3f}} | {{significance.qrag[grover].ndcg@10.significant}} |
| qrag[qaoa] | {{significance.qrag[qaoa].ndcg@10.delta|signed}} | [{{significance.qrag[qaoa].ndcg@10.ci95_low|4f}}, {{significance.qrag[qaoa].ndcg@10.ci95_high|4f}}] | {{significance.qrag[qaoa].ndcg@10.p_value|3f}} | {{significance.qrag[qaoa].ndcg@10.significant}} |
| qrag[kernel+qaoa] | {{significance.qrag[kernel+qaoa].ndcg@10.delta|signed}} | [{{significance.qrag[kernel+qaoa].ndcg@10.ci95_low|4f}}, {{significance.qrag[kernel+qaoa].ndcg@10.ci95_high|4f}}] | {{significance.qrag[kernel+qaoa].ndcg@10.p_value|3f}} | {{significance.qrag[kernel+qaoa].ndcg@10.significant}} |
| qrag[full] | {{significance.qrag[full].ndcg@10.delta|signed}} | [{{significance.qrag[full].ndcg@10.ci95_low|4f}}, {{significance.qrag[full].ndcg@10.ci95_high|4f}}] | {{significance.qrag[full].ndcg@10.p_value|3f}} | {{significance.qrag[full].ndcg@10.significant}} |

[[CAPTION]] Table 5.2: Paired bootstrap, {{config.eval.bootstrap_samples}} resamples, over all {{n_queries}} queries.

**The retrieval result is null.** Every nDCG@10 delta lies within a few thousandths
of the baseline, every confidence interval spans zero, and no arm is significant.
Across the whole grid — {{sig.total}} system-metric cells, {{n_systems}} minus one
systems by five metrics — exactly {{sig.count}} cell reaches p < 0.05:
`qrag[kernel]` on recall@5, at {{sig.first.delta|signed}} with
p = {{sig.first.p|3f}}. With {{sig.total}} comparisons and no correction, that is the
expected yield of chance. We do not present it as a finding.

The interesting part of this null is its relationship to Section 5.1. The kernel
satisfied both criteria that were set in advance, on held-out data, and then did not
transfer to {{n_queries}} queries against a full corpus. That is a clean instance of
the overfitting predicted in Section 5.1 from the ratio of free phases to training
pairs, and it is the reason the criteria were pre-registered: had they been chosen
after the fact, the held-out improvement would have been reported as the headline and
this section would not exist.

## 5.3 Grover: query complexity and simulation cost

| Quantity | Value |
|---|---|
| Shortlist size | {{grover.candidates|0f}} candidates, {{grover.qubits}} qubits |
| Marked items (mean) | {{grover.marked|0f}} |
| Oracle queries, Grover (mean) | {{grover.oracle_queries|2f}} |
| Oracle queries, classical scan (expected) | {{grover.classical_queries|2f}} |
| **Query reduction factor** | **{{grover.reduction|2f}}x** |
| Success probability (mean) | {{grover.success}} |
| **Simulation overhead** | **{{grover.overhead|2f}}x slower than the classical scan** |
| Simulated wall clock, stage only | {{grover.ms|3f}} ms/query |
| Simulation overhead, Grover-only arm | {{grover_arm.overhead|2f}}x ({{grover_arm.ms|3f}} ms/query) |

[[CAPTION]] Table 5.3: All figures from the `qrag[full]` pipeline except the final row, which is the `qrag[grover]` arm. The reduction factor and the overhead are different quantities in different units and neither substitutes for the other.

The {{grover.reduction|2f}}-fold reduction is in oracle queries: Grover needs
{{grover.oracle_queries|2f}} query where a classical scan of the same shortlist needs
{{grover.classical_queries|2f}} in expectation, which for {{grover.marked|0f}} marked
items among {{grover.candidates|0f}} is the textbook (N+1)/(M+1). That is a property
of the algorithm and independent of hardware. The {{grover.overhead|2f}}-fold overhead
is the cost of simulating the algorithm on a classical machine, which is also
expected: a statevector simulator cannot outperform the routine it simulates.
Printing them adjacently is the reporting discipline of Section 3.6, and the reason
for it is that in the applied literature the first number is frequently restated as
though it were a latency result. The last row exists because the Grover-only arm's
overhead ({{grover_arm.overhead|2f}}x) is not the same number as the full pipeline's
({{grover.overhead|2f}}x), and quoting one where the other belongs would be the same
class of error.

Two further observations belong here rather than in a footnote.

First, the Grover arm's retrieval metrics in Table 5.1 are identical to the
baseline's on every metric — recall@10 {{clean.qrag[grover].metrics.recall@10}},
nDCG@10 {{clean.qrag[grover].metrics.ndcg@10}}, MRR@10
{{clean.qrag[grover].metrics.mrr@10}}, and a bootstrap delta of exactly
{{significance.qrag[grover].ndcg@10.delta|signed}} with p =
{{significance.qrag[grover].ndcg@10.p_value|3f}}. This is correct behaviour, not a
bug: the stage searches an already-ranked shortlist, so it cannot change what is
retrieved. It contributes zero to retrieval quality by construction, and we report
that rather than describing the stage as "maintaining baseline quality".

Second, the Grover arm's end-to-end latency in Table 5.1 is
{{clean.qrag[grover].metrics.latency_ms_mean}} ms/query against the baseline's
{{base.ms}} ms/query — that is, marginally *lower* while doing strictly more work.
This is measurement noise on a sub-40 ms path, not a speed-up, and we flag it
explicitly because a {{base.ms}}-to-{{clean.qrag[grover].metrics.latency_ms_mean}}
comparison is exactly the kind of difference that could be quoted as one. The stage's
own cost is {{grover.ms|3f}} ms/query, far below the run-to-run variation of the
retrieval path it sits in.

## 5.4 QAOA: solution quality and cost

| Quantity | Value |
|---|---|
| Qubits, layers | {{qaoa.qubits}}, p = {{qaoa.layers}} |
| Mean solution quality vs exact optimum | {{qaoa.quality}} |
| Worst-case solution quality | {{qaoa.quality_worst}} |
| Queries attaining the exact optimum | {{qaoa.exact_rate|pct}} |
| Mean objective gap | {{clean.qrag[full].quantum.qaoa.mean_objective_gap}} |
| Max objective gap | {{qaoa.max_gap}} |
| Degenerate queries (no choice to make) | {{clean.qrag[full].quantum.qaoa.n_degenerate}} |
| **Feasible probability mass at readout** | **{{qaoa.feasible_prob|2f}}** |
| Optimiser calls per query (mean) | {{qaoa.optimiser_calls|0f}} |
| Mean redundancy, top-k selection | {{qaoa.redundancy_before}} |
| Mean redundancy, QAOA selection | {{qaoa.redundancy_after}} |
| Queries where QAOA was less redundant | {{clean.qrag[full].quantum.qaoa.fraction_less_redundant|pct}} |
| **Simulated wall clock** | **{{qaoa.ms}} ms/query** |

[[CAPTION]] Table 5.4: QAOA against a brute-force exact optimum over the same exactly-k feasible set.

QAOA solves the selection QUBO well: quality {{qaoa.quality}} against the exact
optimum, with the exact optimum attained on {{qaoa.exact_rate|pct}} of queries and a
worst case of {{qaoa.quality_worst}}. No query was degenerate
({{clean.qrag[full].quantum.qaoa.n_degenerate}} of {{n_queries}}), so every quality
figure is a real comparison rather than a division over a flat objective. QAOA also
reduces selected-set redundancy from {{qaoa.redundancy_before}} to
{{qaoa.redundancy_after}}, and does so on
{{clean.qrag[full].quantum.qaoa.fraction_less_redundant|pct}} of queries.

**The quality figure is conditional, and the condition is load-bearing.** The
cardinality constraint enters the QUBO as a soft penalty, so the optimised state
retains amplitude on selections of the wrong size, and only
{{qaoa.feasible_prob|2f}} of the probability mass at readout satisfies the exactly-*k*
constraint. The reported {{qaoa.quality}} is the quality of the most probable
*feasible* selection after that filtering — roughly
{{clean.qrag[full].quantum.qaoa.mean_feasible_probability|2f}} of measurements are
usable and the rest are discarded. On a simulator this is free, because the full
distribution is available by construction. On hardware it is not: it implies on the
order of 1/{{qaoa.feasible_prob|2f}} times as many shots to obtain the same decoded
answer, and that multiplier is absent from every figure in this table. A quality
number quoted without its feasible mass overstates what the routine delivers, so both
appear in the same table.

The cost is the dominant fact about this stage. At {{qaoa.ms}} ms/query it accounts
for {{qaoa.share|pct}} of the pipeline's per-query latency, is {{qaoa.vs_rest|1f}}
times the other four stages combined, and is {{qaoa.vs_base|1f}} times the classical
baseline's entire per-query cost. On a clean corpus it buys no measurable retrieval
gain in exchange (Table 5.2). Whether it earns that cost anywhere is the subject of
Section 5.5, and the answer there is a qualified yes on one axis that is not
retrieval quality.

## 5.5 Adversarial robustness

| System (poisoned corpus) | context occupancy | clean-context rate | top-10 hit rate | nDCG@10 |
|---|---|---|---|---|
| classical-baseline | {{poisoned.systems.classical-baseline.attack.context_occupancy}} | {{poisoned.systems.classical-baseline.attack.clean_context_rate}} | {{poisoned.systems.classical-baseline.attack.top_k_hit_rate}} | {{poisoned.systems.classical-baseline.metrics_poisoned.ndcg@10}} |
| qrag[no-qaoa] | {{poisoned.systems.qrag[no-qaoa].attack.context_occupancy}} | {{poisoned.systems.qrag[no-qaoa].attack.clean_context_rate}} | {{poisoned.systems.qrag[no-qaoa].attack.top_k_hit_rate}} | {{poisoned.systems.qrag[no-qaoa].metrics_poisoned.ndcg@10}} |
| qrag[full] | {{poisoned.systems.qrag[full].attack.context_occupancy}} | {{poisoned.systems.qrag[full].attack.clean_context_rate}} | {{poisoned.systems.qrag[full].attack.top_k_hit_rate}} | {{poisoned.systems.qrag[full].metrics_poisoned.ndcg@10}} |

[[CAPTION]] Table 5.5: {{poison.n_injected}} adversarial passages, {{poison.n_targets}} target queries, {{poison.n_families}} families. Judgments unmodified.

**The positive result.** Adversarial context occupancy falls from
{{poison.occ_noqaoa}} in `qrag[no-qaoa]` to {{poison.occ_full}} in `qrag[full]`, a
reduction of {{poison.qaoa_effect}}. The two arms are identical except for the QAOA
selection stage, so the reduction is attributable to the pairwise redundancy penalty
in the QUBO of Section 3.7. The mechanism is intelligible: injected passages in a
family are mutually similar, so a term that penalises selecting mutually similar
passages selects fewer of them. This is the one place in the paper where a quantum
stage does something a classical top-k selector does not, and it is why the ablation
arm was included.

**What the result does not mean.** Four counterweights belong in the same paragraph
as the number above.

1. {{poison.occ_full}} occupancy is a total compromise, not a defended system. All
   {{poison.n_targets}} targeted queries are hit (top-10 hit rate
   {{poison.hit_rate_full}}) and the clean-context rate is
   {{poison.clean_rate_full}} — not one targeted query received an uncontaminated
   context.
2. Retrieval quality degrades regardless: nDCG@10 on the poisoned corpus falls to
   {{poison.ndcg_full}} for the full pipeline, a change of
   {{poison.ndcg_drop_full|signed}} against its clean-corpus value.
3. The pattern detector flags {{detector.flagged}} of {{detector.total}} injected
   passages, an aggregate of {{detector.rate|pct}}. That aggregate is close to
   meaningless on its own: it is {{detector.instruction-injection|pct}} on
   instruction-injection and {{detector.topical-mimicry|pct}},
   {{detector.lexical-gaming|pct}} and {{detector.embedding-optimised|pct}} on the
   other three families. The detector catches the one family that announces itself
   with imperative phrasing and none of the families that do not. Reporting only the
   aggregate would conceal exactly that.
4. The redundancy penalty is not a security mechanism. It is a diversity objective
   that happens to disfavour clustered injections; an attacker who diversifies
   injected passages should be expected to defeat it, and we did not test that
   attacker.

## 5.6 Cost of the full pipeline

The full pipeline runs at {{full.ms}} ms/query against the baseline's
{{base.ms}} ms/query, a factor of {{slowdown}}. End to end, the baseline sweep takes
{{base.run_seconds|0f}} s and the full pipeline {{full.run_seconds|0f}} s over the
same {{n_queries}} queries. Since {{qaoa.share|pct}} of the pipeline's per-query time
is the QAOA stage, the slowdown is essentially one stage's simulation cost.

We state the obvious interpretation so that it is not left to inference: this is the
cost of classical simulation, it says nothing about what the same circuits would cost
on hardware, and it is not evidence about the algorithms' complexity. It is reported
because a paper that omits it invites the reader to assume the pipeline is
competitive on time, and it is not.

# 6. Discussion

## 6.1 A null result on a hard corpus is a real result

The retrieval outcome is that {{n_systems}} configurations, one of which passed two
pre-registered criteria on held-out data, produced no significant improvement on
{{n_queries}} queries. Three properties make it worth reporting rather than
discarding. The baseline is the same fusion with weight moved onto cosine, so the
comparison isolates the kernel. The corpus is the full test split, not a favourable
sample. And the criteria were set before training, so the pass in Section 5.1 and the
null in Section 5.2 are both on the record, and the gap between them is the finding:
a kernel can clear reasonable acceptance criteria on held-out pairs and still not
transfer.

## 6.2 Where the rank-equivalence argument leaves the field

Section 3.3's measurement is, we think, the most transferable part of this work. It
costs little to run: score a candidate set with the proposed quantum similarity and
with the classical similarity it is meant to improve, and report Kendall tau between
the orderings. If tau is close to 1, the quantum stage is not the cause of any
observed difference, and the burden shifts to identifying what is. We would suggest
this becomes a standard reported diagnostic in the quantum IR literature, precisely
because our own global kernel failed it at {{gate_a.tau_global_before}} and we would
not have known from retrieval metrics alone — the metrics would simply have looked
like the baseline's, which is easy to describe as parity.

## 6.3 The security result, weighed

The honest weighing is that a {{poison.qaoa_effect}} reduction in occupancy, obtained
at {{qaoa.ms}} ms/query and conditional on discarding
{{clean.qrag[full].quantum.qaoa.mean_feasible_probability|2f}} of the readout mass, is
not a good trade for a practitioner. A classical diversity-aware selector such as MMR
(Carbonell and Goldstein, 1998) implements the same objective at negligible cost. Our
own codebase contains one — `rerank_greedy_mmr`, greedily maximising relevance minus
the maximum similarity to the already-selected set, over the identical QUBO — and it
is used in the solver comparison but was **not** run as an arm of the poisoned-corpus
experiment. The missing comparison is therefore a missing run, not a missing
implementation, which makes it both the cheapest and the most important item of
further work: it is the measurement most likely to reduce Section 5.5 from a result
about QAOA to a result about the objective. We state this rather than leaving the
reader to assume the comparison was considered and rejected.

What the arm does establish is narrower and still worth having: the redundancy term,
not the kernel and not the interference rerank, is the component that moves
adversarial occupancy, and it moves it in the direction one would predict from the
objective. That is a mechanism identified by ablation rather than a defence
demonstrated by benchmark.

## 6.4 Threats to validity

**Single corpus.** All results are on SciFact. It was chosen as an unfavourable case,
which strengthens the null result and weakens any generalisation of the security
result.

**Simulation only.** Every quantum stage is simulated at small qubit counts
({{grover.qubits}} and {{qaoa.qubits}}). Nothing here speaks to noise, decoherence,
or the barren-plateau behaviour that afflicts variational training at scale (McClean
et al., 2018).

**One attack budget.** {{poison.n_injected}} passages against
{{poison.n_targets}} queries is one point in a space parameterised by budget,
diversity and query targeting. The occupancy reduction is measured at that point
only.

**Overfitted kernel.** {{n_blocks}} × {{block_size}} free phases against
{{n_train_pairs}} pairs. The transfer failure in Section 5.2 is consistent with
overfitting, but we did not isolate overfitting from the alternative explanation that
the block structure captures nothing retrieval-relevant on this corpus. Separating
those two requires a training-set sweep we have not run.

**Latency methodology.** Single machine, mean over queries, no warm-up isolation
beyond a fixed cache. Adequate for the coarse cost accounting it is used for and not
adequate for anything finer.

# 7. Reproducibility

The experiment of record is a single configuration, hashed and recorded in every
result file, together with the interpreter version, numpy version, platform and seed.
Configuration `{{config_hash}}`, seed `{{seed}}`, Python {{python}}, numpy
{{numpy}}: the same command on the same corpus reproduces the numbers in this paper.

Every figure in this paper, in the accompanying slide deck and in the project diary
is substituted from the result files at document build time rather than typed. Tokens
resolve against `results/experiment.json`, `results/kernel_training.json` and
`results/verification.json`, and an unrecognised token fails the build rather than
rendering blank. The purpose is narrow and practical: a figure typed into several
documents can disagree with itself, and the reader most likely to notice is an
examiner. The mechanism has already caught four stale figures in earlier drafts of
this project's prose, including one place where the whole pipeline's per-query
latency had been attributed to a single stage.

Verification is likewise read rather than remembered: {{tests.passed}} tests pass
({{tests.failed}} failures), and the security audit reports {{audit.passed}} checks
passed, {{audit.failed}} failed, {{audit.warned}} warned and {{audit.na}} not
applicable.

# 8. Conclusion and further work

> **Status note.** This section is a draft. The empirical sections above (3 through
> 7) are complete against the experiment of record; the items below are scoped but
> not executed, and the generation-stage evaluation in particular is not yet built.
> This paper is submitted as partially complete for that reason.

We built a RAG pipeline with three simulated quantum stages and measured it against a
baseline designed to be hard to beat. The retrieval result is null: {{sig.count}} of
{{sig.total}} comparisons significant, which is chance. The cost result is a
{{slowdown}}-fold slowdown, which is expected under simulation. The one positive
result is a {{poison.qaoa_effect}} reduction in adversarial context occupancy
attributable by ablation to the QAOA redundancy penalty, reported alongside the fact
that the resulting system is still entirely compromised.

The most useful transferable contribution is probably the smallest: a fidelity kernel
should be checked for rank-equivalence against the classical similarity it replaces,
by measuring Kendall tau between the two orderings, before any retrieval experiment
is run on it. Our global kernel measured {{gate_a.tau_global_before}} on that test.

Work remaining, in the order we would do it:

1. **The missing classical arm.** `rerank_greedy_mmr` at the same *k* and the same
   redundancy weight, run on the poisoned corpus. The solver already exists; only the
   experiment arm is missing. If MMR achieves a comparable occupancy reduction at
   negligible cost — which we expect — then Section 5.5's result is a statement about
   the objective, not about QAOA, and should be rewritten as such.
2. **Shot accounting for the feasibility filter.** The {{qaoa.quality}} quality figure
   post-selects on the {{qaoa.feasible_prob|2f}} of readout mass that satisfies the
   cardinality constraint. A hard-constrained ansatz, or an XY-mixer restricted to the
   fixed-Hamming-weight subspace, would remove the filter; failing that, the shot
   multiplier belongs in the cost accounting of Table 5.4.
3. **Generation-stage evaluation.** The pipeline retrieves and selects but the
   end-to-end answer-generation and answer-relevancy evaluation (Es et al., 2024;
   Chen et al., 2024) is not built. Until it is, "RAG" describes the architecture and
   not the measurement, and downstream faithfulness under poisoning is unmeasured.
4. **A second corpus.** At least one with a weaker lexical baseline, to test whether
   the null result is a property of SciFact.
5. **A diversified attacker.** Injections spread across topics and phrasings, to test
   the prediction in Section 5.5 that the redundancy penalty fails against them.
6. **A training-set sweep** on the kernel, to separate overfitting from the
   hypothesis that the block structure carries no retrieval-relevant signal here.

# References

Aaronson, S. (2015). Read the fine print. *Nature Physics*, 11(4), 291-293.

Bennett, C. H., Bernstein, E., Brassard, G., and Vazirani, U. (1997). Strengths and
weaknesses of quantum computing. *SIAM Journal on Computing*, 26(5), 1510-1523.

Bharti, K., Cervera-Lierta, A., Kyaw, T. H., Haug, T., Alperin-Lea, S., Anand, A.,
Degroote, M., Heimonen, H., Kottmann, J. S., Menke, T., Mok, W.-K., Sim, S., Kwek,
L.-C., and Aspuru-Guzik, A. (2022). Noisy intermediate-scale quantum algorithms.
*Reviews of Modern Physics*, 94(1), 015004.

Boyer, M., Brassard, G., Hoyer, P., and Tapp, A. (1998). Tight bounds on quantum
searching. *Fortschritte der Physik*, 46(4-5), 493-505.

Brassard, G., Hoyer, P., Mosca, M., and Tapp, A. (2002). Quantum amplitude
amplification and estimation. *Contemporary Mathematics*, 305, 53-74.

Bravyi, S., Kliesch, A., Koenig, R., and Tang, E. (2020). Obstacles to variational
quantum optimization from symmetry protection. *Physical Review Letters*, 125(26),
260505.

Carbonell, J. and Goldstein, J. (1998). The use of MMR, diversity-based reranking for
reordering documents and producing summaries. In *Proceedings of the 21st Annual
International ACM SIGIR Conference on Research and Development in Information
Retrieval*, 335-336.

Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K.,
McClean, J. R., Mitarai, K., Yuan, X., Cincio, L., and Coles, P. J. (2021).
Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625-644.

Chen, J., Lin, H., Han, X., and Sun, L. (2024). Benchmarking large language models in
retrieval-augmented generation. In *Proceedings of the AAAI Conference on Artificial
Intelligence*, 38(16), 17754-17762.

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. (2024). BGE
M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings
through self-knowledge distillation. In *Findings of the Association for
Computational Linguistics: ACL 2024*.

Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal rank fusion
outperforms Condorcet and individual rank learning methods. In *Proceedings of the
32nd International ACM SIGIR Conference on Research and Development in Information
Retrieval*, 758-759.

Ebrahimi, J., Rao, A., Lowd, D., and Dou, D. (2018). HotFlip: White-box adversarial
examples for text classification. In *Proceedings of the 56th Annual Meeting of the
Association for Computational Linguistics (ACL)*, 31-36.

Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman
and Hall, New York.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated
evaluation of retrieval augmented generation. In *Proceedings of the 18th Conference
of the European Chapter of the Association for Computational Linguistics (EACL):
System Demonstrations*, 150-158.

Farhi, E., Goldstone, J., and Gutmann, S. (2014). A quantum approximate optimization
algorithm. *arXiv preprint arXiv:1411.4028*.

Glover, F., Kochenberger, G., and Du, Y. (2019). A tutorial on formulating and using
QUBO models. *4OR: A Quarterly Journal of Operations Research*, 17, 335-371.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., and Fritz, M. (2023).
Not what you've signed up for: Compromising real-world LLM-integrated applications
with indirect prompt injection. In *Proceedings of the 16th ACM Workshop on
Artificial Intelligence and Security (AISec)*, 79-90.

Grover, L. K. (1996). A fast quantum mechanical algorithm for database search. In
*Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC)*,
212-219.

Harrow, A. W., Hassidim, A., and Lloyd, S. (2009). Quantum algorithm for linear
systems of equations. *Physical Review Letters*, 103(15), 150502.

Hastings, M. B. (2019). Classical and quantum bounded depth approximation algorithms.
*arXiv preprint arXiv:1905.07047*.

Havlicek, V., Corcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M.,
and Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces.
*Nature*, 567(7747), 209-212.

Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H., and
McClean, J. R. (2021). Power of data in quantum machine learning. *Nature
Communications*, 12(1), 2631.

Jarvelin, K. and Kekalainen, J. (2002). Cumulated gain-based evaluation of IR
techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.

Javadi-Abhari, A., Treinish, M., Krsulich, K., Wood, C. J., Lishman, J., Gacon, J.,
Martiel, S., Nation, P. D., Bishop, L. S., Cross, A. W., Johnson, B. R., and
Gambetta, J. M. (2024). Quantum computing with Qiskit. *arXiv preprint
arXiv:2405.08810*.

Johnson, J., Douze, M., and Jegou, H. (2021). Billion-scale similarity search with
GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih,
W.-t. (2020). Dense passage retrieval for open-domain question answering. In
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP)*, 6769-6781.

Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*, 30(1-2),
81-93.

Kerenidis, I. and Prakash, A. (2017). Quantum recommendation systems. In *Proceedings
of the 8th Innovations in Theoretical Computer Science Conference (ITCS)*,
49:1-49:21.

Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. In
*Proceedings of the International Conference on Learning Representations (ICLR)*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H.,
Lewis, M., Yih, W.-t., Rocktaschel, T., Riedel, S., and Kiela, D. (2020).
Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in
Neural Information Processing Systems (NeurIPS)*, 33, 9459-9474.

Li, Q., Wang, B., and Melucci, M. (2019). CNM: An interpretable complex-valued
network for matching. In *Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics (NAACL-HLT)*, 4139-4148.

Liu, Y., Arunachalam, S., and Temme, K. (2021). A rigorous and robust quantum
speed-up in supervised machine learning. *Nature Physics*, 17(9), 1013-1017.

Lucas, A. (2014). Ising formulations of many NP problems. *Frontiers in Physics*, 2,
5.

McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., and Neven, H. (2018).
Barren plateaus in quantum neural network training landscapes. *Nature
Communications*, 9(1), 4812.

OWASP (2025). *OWASP Top 10 for Large Language Model Applications*. Open Worldwide
Application Security Project.

Perez, F. and Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for
language models. In *NeurIPS ML Safety Workshop*.

Piwowarski, B., Frommholz, I., Lalmas, M., and van Rijsbergen, K. (2010). What can
quantum theory bring to information retrieval? In *Proceedings of the 19th ACM
International Conference on Information and Knowledge Management (CIKM)*, 59-68.

Powell, M. J. D. (1994). A direct search optimization method that models the
objective and constraint functions by linear interpolation. In *Advances in
Optimization and Numerical Analysis*, 51-67. Springer, Dordrecht.

Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, 2, 79.

Robertson, S. and Zaragoza, H. (2009). The probabilistic relevance framework: BM25
and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

Sakai, T. (2006). Evaluating evaluation metrics based on the bootstrap. In
*Proceedings of the 29th Annual International ACM SIGIR Conference on Research and
Development in Information Retrieval*, 525-532.

Schuld, M. (2021). Supervised quantum machine learning models are kernel methods.
*arXiv preprint arXiv:2101.11020*.

Schuld, M. and Killoran, N. (2019). Quantum machine learning in feature Hilbert
spaces. *Physical Review Letters*, 122(4), 040504.

Sordoni, A., Nie, J.-Y., and Bengio, Y. (2013). Modeling term dependencies with
quantum language models for IR. In *Proceedings of the 36th International ACM SIGIR
Conference on Research and Development in Information Retrieval*, 653-662.

Tang, E. (2019). A quantum-inspired classical algorithm for recommendation systems.
In *Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing
(STOC)*, 217-228.

Thakur, N., Reimers, N., Ruckle, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A
heterogeneous benchmark for zero-shot evaluation of information retrieval models. In
*Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks
Track*.

Uprety, S., Gkoumas, D., and Song, D. (2020). A survey of quantum theory inspired
approaches to information retrieval. *ACM Computing Surveys*, 53(5), 1-39.

van den Oord, A., Li, Y., and Vinyals, O. (2018). Representation learning with
contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

van Rijsbergen, C. J. (2004). *The Geometry of Information Retrieval*. Cambridge
University Press, Cambridge.

Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., and Hajishirzi,
H. (2020). Fact or fiction: Verifying scientific claims. In *Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 7534-7550.

Wallace, E., Feng, S., Kandpal, N., Gardner, M., and Singh, S. (2019). Universal
adversarial triggers for attacking and analyzing NLP. In *Proceedings of the 2019
Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2153-2162.

Wang, B., Li, Q., Melucci, M., and Song, D. (2019). Semantic Hilbert space for text
representation learning. In *Proceedings of the World Wide Web Conference (WWW)*,
3293-3299.

Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P., Ahmed, J., and
Overwijk, A. (2021). Approximate nearest neighbor negative contrastive learning for
dense text retrieval. In *Proceedings of the International Conference on Learning
Representations (ICLR)*.

Zhang, P., Niu, J., Su, Z., Wang, B., Ma, L., and Song, D. (2018). End-to-end
quantum-like language models with application to question answering. In *Proceedings
of the AAAI Conference on Artificial Intelligence*, 32(1), 5666-5673.

Zhong, Z., Huang, Z., Wettig, A., and Chen, D. (2023). Poisoning retrieval corpora by
injecting adversarial passages. In *Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing (EMNLP)*, 13764-13775.

Zhou, L., Wang, S.-T., Choi, S., Pichler, H., and Lukin, M. D. (2020). Quantum
approximate optimization algorithm: Performance, mechanism, and implementation on
near-term devices. *Physical Review X*, 10(2), 021067.

Zou, W., Geng, R., Wang, B., and Jia, J. (2025). PoisonedRAG: Knowledge corruption
attacks to retrieval-augmented generation of large language models. In *Proceedings
of the 34th USENIX Security Symposium*.
