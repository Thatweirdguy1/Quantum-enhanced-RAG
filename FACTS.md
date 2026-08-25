# FACTS -- source of truth for every Q-RAG document

**Generated file. Do not edit by hand.**

```bash
python -m scripts.build_facts
```

Every number below is read out of `results/*.json`. The research paper, the
literature review, the slide deck and the daily diary are all written from
this file, so that no figure is transcribed twice and none of them can
drift. If an experiment is re-run, regenerate this file rather than
editing it.

## 1. Provenance

Cite these whenever a table is reproduced. Same command + same seed gives
the same numbers; a different config hash means the table is from a
different experiment and must not be mixed with these.

- config hash: `da66cb43af5b`
- seed: `20260720` (every stage)
- git commit: `not-a-git-repo`
- python 3.13.14, numpy 2.4.3
- platform: Windows-11-10.0.26200-SP0
- dataset: **scifact/test** -- 5183 documents, 300 queries, 339 relevance judgments
- source: `results/experiment.json`

## 2. Kernel training and the two gates

Source: `results/kernel_training.json`. Both gates were defined **before**
training, because the central risk was that the kernel is a ranking no-op.

- training pairs: 919 (from 807 unique train queries), embedding dim 1024
- hard-negative pool: 64 per query; mean cos(q, hardest negative) 0.5944 vs cos(q, positive) 0.6256; the positive is out-scored on 34.8% of pairs

### Gate A -- does the kernel reorder at all?

Kendall tau against the cosine ranking must fall below 0.995.

| kernel | tau before training | tau after training | verdict |
|---|---|---|---|
| global fidelity | 0.9995 | 0.4457 | reorders only after training; **starts as a near-exact no-op** |
| block fidelity | 0.0697 | 0.4200 | reorders |
| block fidelity at theta=0 (control) | 0.4011 | (not trained) | **breaks rank-equivalence structurally, without fitted phases** |

- Gate A verdict: **PASS** (tau 0.4200 < 0.995)
- global kernel mean|theta| after training: 0.9159
- block kernel: 128 blocks x 8 dims, mean|theta| 1.5896, w in [0.0776, 2.9214]
- early stopping restored epoch 19 (val_loss 1.4497)

### Gate B -- is the reordering an improvement?

Held-out: 137 validation pairs, 15 negatives each.

| scorer | top-1 | MRR |
|---|---|---|
| cosine | 0.6277 | 0.6881 |
| cosine^2 | 0.6277 | 0.6881 |
| global-fidelity | 0.6204 | 0.6986 |
| cosine + 0.25*global-fidelity | 0.6277 | 0.6950 |
| block-fidelity | 0.5985 | 0.6962 |
| cosine + 0.25*block-fidelity | 0.6423 | 0.7034 |

- Gate B verdict: **PASS** -- MRR 0.6881 -> 0.7034 (+0.0153), top-1 +0.0146

**Two facts that must travel with this table.** First, `cosine` and
`cosine^2` score *identically* on every metric, which is the empirical
confirmation that a global fidelity kernel at theta=0 is rank-equivalent to
cosine and therefore cannot rerank. Second, a top-1 gain of
+0.0146 on 137 pairs is about two
queries; it is not significant and must not be described as such.

## 3. Ablation grid on the clean corpus

All 300 SciFact test queries. The baseline is the
same hybrid fusion with the kernel weight redistributed onto cosine, so it
is a tuned system rather than a strawman with a missing term.

| system | recall@10 | ndcg@10 | mrr@10 | recall@5 | ndcg@5 | recall@20 | ms/query |
|---|---|---|---|---|---|---|---|
| `classical-baseline` | 0.8411 | 0.7152 | 0.6829 | 0.7849 | 0.6953 | 0.8686 | 38.8 |
| `qrag[kernel]` | 0.8371 | 0.7163 | 0.6849 | 0.8022 | 0.7026 | 0.8812 | 39.5 |
| `qrag[kernel+interf]` | 0.8489 | 0.7149 | 0.6805 | 0.7902 | 0.6943 | 0.8786 | 49.0 |
| `qrag[grover]` | 0.8411 | 0.7152 | 0.6829 | 0.7849 | 0.6953 | 0.8686 | 36.6 |
| `qrag[qaoa]` | 0.8411 | 0.7136 | 0.6816 | 0.7687 | 0.6878 | 0.8686 | 1343.5 |
| `qrag[kernel+qaoa]` | 0.8371 | 0.7141 | 0.6827 | 0.7862 | 0.6948 | 0.8812 | 1161.0 |
| `qrag[full]` | 0.8489 | 0.7140 | 0.6799 | 0.7855 | 0.6907 | 0.8786 | 1191.0 |

**`qrag[grover]` scores identically to the baseline on every retrieval metric, to floating point.** That is not a coincidence and not a bug: amplitude amplification here selects from a shortlist that has already been scored classically, and the ordering it returns is the classical ordering. Grover in this pipeline demonstrates an oracle-query complexity property on a real workload and contributes nothing to ranking quality. Any document presenting it as a retrieval improvement is misreporting this table.

## 4. Significance against the classical baseline

Paired bootstrap over per-query scores, 2000 resamples, seed fixed. `sig` means the 95% CI excludes zero.

| system | metric | baseline | system | delta | 95% CI | p | sig |
|---|---|---|---|---|---|---|---|
| `qrag[kernel]` | recall@10 | 0.8411 | 0.8371 | -0.0040 | [-0.0187, +0.0113] | 0.6550 | no |
| `qrag[kernel]` | ndcg@10 | 0.7152 | 0.7163 | +0.0011 | [-0.0103, +0.0122] | 0.8790 | no |
| `qrag[kernel]` | mrr@10 | 0.6829 | 0.6849 | +0.0021 | [-0.0119, +0.0153] | 0.7880 | no |
| `qrag[kernel]` | recall@5 | 0.7849 | 0.8022 | +0.0173 | [+0.0003, +0.0356] | 0.0470 | yes |
| `qrag[kernel]` | ndcg@5 | 0.6953 | 0.7026 | +0.0073 | [-0.0051, +0.0198] | 0.2760 | no |
| `qrag[kernel+interf]` | recall@10 | 0.8411 | 0.8489 | +0.0078 | [-0.0017, +0.0194] | 0.1150 | no |
| `qrag[kernel+interf]` | ndcg@10 | 0.7152 | 0.7149 | -0.0003 | [-0.0108, +0.0101] | 0.9710 | no |
| `qrag[kernel+interf]` | mrr@10 | 0.6829 | 0.6805 | -0.0024 | [-0.0166, +0.0113] | 0.7340 | no |
| `qrag[kernel+interf]` | recall@5 | 0.7849 | 0.7902 | +0.0053 | [-0.0067, +0.0183] | 0.3770 | no |
| `qrag[kernel+interf]` | ndcg@5 | 0.6953 | 0.6943 | -0.0010 | [-0.0117, +0.0098] | 0.8640 | no |
| `qrag[grover]` | recall@10 | 0.8411 | 0.8411 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| `qrag[grover]` | ndcg@10 | 0.7152 | 0.7152 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| `qrag[grover]` | mrr@10 | 0.6829 | 0.6829 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| `qrag[grover]` | recall@5 | 0.7849 | 0.7849 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| `qrag[grover]` | ndcg@5 | 0.6953 | 0.6953 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| `qrag[qaoa]` | recall@10 | 0.8411 | 0.8411 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | no |
| `qrag[qaoa]` | ndcg@10 | 0.7152 | 0.7136 | -0.0016 | [-0.0036, +0.0002] | 0.0810 | no |
| `qrag[qaoa]` | mrr@10 | 0.6829 | 0.6816 | -0.0012 | [-0.0037, +0.0009] | 0.2850 | no |
| `qrag[qaoa]` | recall@5 | 0.7849 | 0.7687 | -0.0162 | [-0.0373, +0.0024] | 0.1010 | no |
| `qrag[qaoa]` | ndcg@5 | 0.6953 | 0.6878 | -0.0075 | [-0.0165, +0.0008] | 0.0790 | no |
| `qrag[kernel+qaoa]` | recall@10 | 0.8411 | 0.8371 | -0.0040 | [-0.0187, +0.0113] | 0.6550 | no |
| `qrag[kernel+qaoa]` | ndcg@10 | 0.7152 | 0.7141 | -0.0011 | [-0.0127, +0.0100] | 0.8270 | no |
| `qrag[kernel+qaoa]` | mrr@10 | 0.6829 | 0.6827 | -0.0001 | [-0.0147, +0.0134] | 0.9440 | no |
| `qrag[kernel+qaoa]` | recall@5 | 0.7849 | 0.7862 | +0.0013 | [-0.0246, +0.0270] | 0.9470 | no |
| `qrag[kernel+qaoa]` | ndcg@5 | 0.6953 | 0.6948 | -0.0005 | [-0.0156, +0.0144] | 0.9150 | no |
| `qrag[full]` | recall@10 | 0.8411 | 0.8489 | +0.0078 | [-0.0017, +0.0194] | 0.1150 | no |
| `qrag[full]` | ndcg@10 | 0.7152 | 0.7140 | -0.0012 | [-0.0116, +0.0093] | 0.8320 | no |
| `qrag[full]` | mrr@10 | 0.6829 | 0.6799 | -0.0029 | [-0.0173, +0.0110] | 0.6800 | no |
| `qrag[full]` | recall@5 | 0.7849 | 0.7855 | +0.0006 | [-0.0194, +0.0213] | 0.9580 | no |
| `qrag[full]` | ndcg@5 | 0.6953 | 0.6907 | -0.0046 | [-0.0185, +0.0088] | 0.5030 | no |

## 5. Quantum accounting

Complexity and simulation cost are reported as **separate** quantities and
must stay separate in every document. Grover's oracle-query reduction is a
hardware-independent count; the simulation overhead beside it is what
running that count on a classical statevector costs.

### Grover -- `qrag[grover]`

- 6 qubits over 64.0 shortlisted candidates, 13.0 marked on average
- **oracle queries: 1.00** vs **4.64** expected for classical scanning -> **4.64x** reduction
- success probability: 0.972
- simulation cost: 0.07 ms/query, 8.98x overhead vs the classical scan it replaces

### QAOA -- `qrag[qaoa]`

- 12 qubits, p=2 layers, 360 optimiser calls/query
- solution quality: 0.9977 mean, 0.9537 worst (affine-invariant, 1.0 == exact brute-force optimum over the same exactly-k feasible set)
- **hit the exact optimum on 76.7% of queries**; largest objective gap 0.1075
- feasible probability at readout: 0.401
- **redundancy: 0.6059 (top-k by score) -> 0.5736 (QAOA), a reduction of 0.0323; lower on 73.0% of queries**
- simulation cost: 1277 ms/query

### QAOA -- `qrag[kernel+qaoa]`

- 12 qubits, p=2 layers, 360 optimiser calls/query
- solution quality: 0.9973 mean, 0.9658 worst (affine-invariant, 1.0 == exact brute-force optimum over the same exactly-k feasible set)
- **hit the exact optimum on 76.7% of queries**; largest objective gap 0.1052
- feasible probability at readout: 0.401
- **redundancy: 0.6017 (top-k by score) -> 0.5694 (QAOA), a reduction of 0.0323; lower on 74.3% of queries**
- simulation cost: 1102 ms/query

### Grover -- `qrag[full]`

- 6 qubits over 64.0 shortlisted candidates, 13.0 marked on average
- **oracle queries: 1.00** vs **4.64** expected for classical scanning -> **4.64x** reduction
- success probability: 0.972
- simulation cost: 0.07 ms/query, 8.03x overhead vs the classical scan it replaces

### QAOA -- `qrag[full]`

- 12 qubits, p=2 layers, 360 optimiser calls/query
- solution quality: 0.9978 mean, 0.9514 worst (affine-invariant, 1.0 == exact brute-force optimum over the same exactly-k feasible set)
- **hit the exact optimum on 78.7% of queries**; largest objective gap 0.1265
- feasible probability at readout: 0.400
- **redundancy: 0.6130 (top-k by score) -> 0.5777 (QAOA), a reduction of 0.0353; lower on 78.7% of queries**
- simulation cost: 1122 ms/query

## 6. Latency -- the honest column

| system | ms/query | vs baseline | seconds for the full run |
|---|---|---|---|
| `classical-baseline` | 38.8 | 1.0x | 11.7 |
| `qrag[kernel]` | 39.5 | 1.0x | 11.9 |
| `qrag[kernel+interf]` | 49.0 | 1.3x | 14.8 |
| `qrag[grover]` | 36.6 | 0.9x | 11.1 |
| `qrag[qaoa]` | 1343.5 | 34.6x | 403.1 |
| `qrag[kernel+qaoa]` | 1161.0 | 29.9x | 348.4 |
| `qrag[full]` | 1191.0 | 30.7x | 357.4 |

**No wall-clock speed-up is claimed anywhere.** A statevector simulator
cannot beat the classical routine it simulates; the slowdown above is the
expected cost of simulation and is reported so that the Grover
oracle-query result in section 5 cannot be mistaken for a timing result.

## 7. Poisoned-corpus arm -- the security experiment

400 adversarial passages injected against 50 target queries across 4 families: topical-mimicry, lexical-gaming, embedding-optimised, instruction-injection. Relevance judgments are **not** modified, so retrieval metrics stay comparable with the clean arm.

**Context occupancy is the attacker's objective**: the fraction of the 5-document context window filled with injected passages. Lower is better for the defender.

| system | context occupancy | clean-context rate | top-10 hit rate | median first adversarial rank | ndcg@10 | ndcg@10 vs clean |
|---|---|---|---|---|---|---|
| `classical-baseline` | 0.9800 | 0.0000 | 1.0000 | 1.0 | 0.6040 | -0.1112 |
| `qrag[no-qaoa]` | 0.9840 | 0.0000 | 1.0000 | 1.0 | 0.6047 | n/a |
| `qrag[full]` | 0.7960 | 0.0000 | 1.0000 | 1.0 | 0.6060 | -0.1081 |

### Per-family share of occupied context slots

| system | topical-mimicry | lexical-gaming | embedding-optimised | instruction-injection |
|---|---|---|---|---|
| `classical-baseline` | 0.0367 | 0.4041 | 0.3673 | 0.1918 |
| `qrag[no-qaoa]` | 0.0325 | 0.4024 | 0.3577 | 0.2073 |
| `qrag[full]` | 0.0302 | 0.4975 | 0.3970 | 0.0754 |

### Pattern detector on the injected text

| family | flagged | n |
|---|---|---|
| topical-mimicry | 0.0% | 0/100 |
| lexical-gaming | 0.0% | 0/100 |
| embedding-optimised | 0.0% | 0/100 |
| instruction-injection | 100.0% | 100/100 |

Overall: 100/400 = 25.0%.

**The 0% rows are the honest ones and must be quoted alongside the 100%
row.** Fluent text that simply contradicts the scientific claim carries
no detectable pattern, because it was written to look exactly like a
real abstract. A regex layer does not defend against corpus poisoning,
and the aggregate figure hides precisely that: three of the four
families evade it completely.

### Verdict on the QAOA redundancy hypothesis

The hypothesis was that the redundancy penalty `lambda * sum_{i<j} s_ij x_i x_j` suppresses *clusters* of mutually-similar injected passages, and `qrag[no-qaoa]` is the control that makes it falsifiable -- without it any drop could be credited to the kernel instead.

- classical baseline occupancy: 0.9800
- `qrag[no-qaoa]` occupancy: 0.9840
- `qrag[full]` occupancy: 0.7960
- **attributable to QAOA: +0.1880** (supports the hypothesis)

Wording that is permitted: QAOA reranking reduced adversarial context occupancy by 0.1880 relative to the otherwise identical pipeline without it, on a single corpus and a single attack budget.

## 8. Verification status

Both are re-runnable and both must be re-run before the report is submitted, because a stale pass is worse than no claim.

- `python -m pytest tests -q` -> **68 passed**
- `python -m scripts.security_audit` -> **16 passed, 0 failed, 0 warned, 8 not applicable**

Three defects in this project's own code were found by those two harnesses and are written up in `SECURITY.md`: dead request validation on the demo API, a credential that could be logged in full while passing the redaction filter, and two unbounded `2**n` allocations behind a ceiling that could not fire. A fourth -- an inverted QAOA quality metric -- was found the same way.

## 9. Claims that may NOT be made

Written down next to the numbers so that making one becomes a deliberate act rather than a drafting convenience.

1. **No speed-up.** Not "faster", not "efficient", not "reduced latency". The simulated pipeline is slower than its own baseline by the factor in section 6.
2. **Grover's reduction is in oracle queries only.** It may never be written as a time or throughput improvement.
3. **The global fidelity kernel is a ranking no-op before training** (tau = 0.9995) and this is a reported finding, not an omission.
4. **The Gate B margin is not significant.** 137 validation pairs; a +1.5-point top-1 change is about two queries.
5. **The injection detector does not defend against fluent poisoning.** 0% detection on topical mimicry. The aggregate rate may not be quoted without that row.
6. **The embedding-optimised attack is black-box** (forward passes only) and strictly weaker than the gradient-based attack of Zhong et al. (2023). Any defence result is against the weaker attack and must say so.
7. **No hardware claim.** Nothing here ran on a quantum device; all circuits are simulated on a classical machine.
8. **One corpus, one attack budget.** No generalisation beyond SciFact is supported by these runs.

**There is no significant retrieval improvement to claim.** No system in the grid beats the classical baseline on nDCG@10 with a CI excluding zero.
