# Q-RAG — Quantum-Enhanced Retrieval-Augmented Generation

Semester-7 major project, Group 165, B.Tech CSE, Amity School of Engineering &
Technology.

A hybrid retrieval pipeline that adds three quantum-simulated stages to a
conventional RAG system, and — more importantly — a set of experiments designed so
that each stage can be shown to be *doing nothing* if that is the truth.

> **Honest summary up front.** This project simulates quantum circuits on a
> classical machine with numpy and qiskit-aer. It cannot and does not claim a
> speed-up: the full pipeline is **30.7× slower** than its own classical baseline.
> And on the headline question it set out to answer, the result is **negative** —
> across 300 SciFact queries and seven system configurations, no quantum arm beats
> the tuned classical baseline on nDCG@10 with a confidence interval excluding zero.
> What the experiments *do* establish is (a) that the kernel reorders results and
> that the reordering is not an improvement at corpus scale, (b) a **4.64× reduction
> in oracle queries** for Grover, which is a hardware-independent quantity, and (c)
> that QAOA's redundancy penalty **cuts adversarial context occupancy from 0.984 to
> 0.796** on a poisoned corpus. Where a result is negative, it is reported as
> negative — see [FACTS.md](FACTS.md), which is generated from the result files and
> ends with a list of claims this project may not make.

---

## What is actually in here

| Stage | What it does | Honest status |
|---|---|---|
| **Hybrid fusion** | BM25 + dense cosine + quantum kernel, per-query min-max normalised, weighted | Working. The classical baseline is the *same* fusion with the kernel weight redistributed onto cosine, so it is a tuned hybrid rather than a strawman |
| **Phase kernel** | Block-local fidelity kernel `K = Σ_g w_g |Σ_{i∈g} qᵢdᵢe^{iθᵢ}|²`, trained with InfoNCE and hand-derived analytic gradients (no autodiff) | Trained and passes both gates — see below. Overfits: 1024 free phases against 919 training pairs |
| **Interference reranking** | Sub-query decomposition with decaying weights | Working, small effect |
| **Grover amplification** | Threshold oracle over a pre-scored shortlist | Working. Reported as **oracle queries**, never as wall clock |
| **QAOA reranking** | Selection QUBO `min −Σrᵢxᵢ + λΣ_{i<j}sᵢⱼxᵢxⱼ + μ(Σxᵢ−k)²`, COBYLA, 3 restarts, exact brute-force optimum computed alongside for a true quality figure | Working, and the most expensive stage by far (~1.1 s/query simulated) |
| **Adversarial arm** | Four attack families injected into the corpus, `qrels` untouched | Working, and the detector's blind spot is measured rather than hidden |
| **Demo API** | FastAPI, hardened to the supplied pre-deploy checklist | Working, deliberately minimal. See [SECURITY.md](SECURITY.md) |

### The two gates

The central risk in this project is that the quantum kernel is a **ranking no-op**.
The global fidelity kernel `|Σᵢqᵢdᵢe^{iθᵢ}|²` at `θ=0` equals `cos²`, which is
rank-identical to cosine — so an untrained kernel changes nothing, and a "quantum
RAG" whose quantum part does not reorder anything is vacuous. Two gates were defined
*before* training and both are checked by `scripts/train_kernel.py`:

- **Gate A — does the kernel reorder at all?** Kendall τ against the cosine ranking
  must be below 0.995. Measured: **τ = 0.420** for the trained block kernel (the
  `θ=0` control sits at 0.401, and the *global* kernel starts at τ = 0.99948, which
  is the no-op it was predicted to be).
- **Gate B — is the reordering an improvement?** Fused MRR must be ≥ cosine MRR on
  held-out pairs. Measured: **0.6881 → 0.7034**.

The block kernel is not merely a trained variant of the global one — it breaks
cosine-equivalence *structurally*. At `θ=0` it equals `Σ_g S_g²`, which by
Cauchy–Schwarz is **not** monotone in `(Σ_g S_g)² = cos²`. That is why it can
reorder where the global kernel cannot.

One correction worth recording: an earlier structural claim of mine — that the global
kernel would collapse to `θ≈0` under training — was **falsified by my own
experiment** (it moved to mean|θ| = 0.916 and its τ fell to 0.446). The retraction is
in the `qrag/kernel.py` docstring rather than quietly dropped.

The Gate B margin is small: **+1.5 points of top-1 on 137 validation pairs is about
two queries.** It is not statistically significant and is not presented as such.

---

## What the experiments measured

Full detail, generated from `results/*.json`, is in [FACTS.md](FACTS.md). The four
results worth stating here:

**1. Retrieval quality: null.** 300 queries, 7 configurations, paired bootstrap with
2,000 resamples. Every nDCG@10 delta against the tuned baseline lies within ±0.002
and none is significant. One cell of 30 reaches p < 0.05 (`qrag[kernel]` recall@5,
+0.0173) and with 30 comparisons that is what noise looks like — it is not claimed as
a finding. The kernel passed both pre-registered gates on held-out pairs and then
did not transfer to the full corpus. That is the result.

**2. Grover: 4.64× fewer oracle queries, 8.98× simulation overhead.** Both numbers
are reported in adjacent columns because the first is a complexity property and the
second is what simulating it costs. `qrag[grover]` scores *identically* to the
baseline on every retrieval metric to floating point — amplitude amplification here
selects from an already-scored shortlist and returns the classical ordering, so it
contributes nothing to ranking quality and is not presented as if it does.

**3. QAOA: 0.9978 mean solution quality, exact optimum on 78.7% of queries** against
brute force over the same feasible set, at 1,122 ms/query for the QAOA stage alone
(1,191 ms/query for the whole pipeline). It is the most expensive stage by a factor
of thirty and buys no retrieval gain on a clean corpus.

**4. Security: the one positive result, and it is narrow.** On a corpus poisoned with
400 passages across four attack families, adversarial context occupancy falls from
**0.984** (`qrag[no-qaoa]`) to **0.796** (`qrag[full]`) — a 0.188 reduction
attributable to the redundancy penalty, isolated by an ablation that differs in that
term alone. The honest framing matters: 0.796 is still catastrophic. Every one of
the 50 targeted queries was hit, no query received a clean context, and the pattern
detector caught **100% of instruction-injection and 0% of the other three families**.
QAOA measurably reduced the attacker's yield and did not come close to defeating the
attack.

---


## Setup

Requires Python 3.11+ and a running [Ollama](https://ollama.com) daemon.

```bash
pip install -r requirements.txt
```

```bash
ollama pull bge-m3
```

```bash
cp .env.example .env
```

Everything is configured through environment variables — there are no credentials in
the source tree, and `scripts/security_audit.py` fails if any appear. Set
`QRAG_LOG_SALT` to a random string; it salts the query fingerprints that appear in
logs in place of the queries themselves.

**Secret rotation warning.** If a secret was ever committed, deleting it in a later
commit does **not** remove it — the old blob is still reachable in history. Rotate it
at the provider, then rewrite history with `git filter-repo` or BFG.

An embedding backend of `hashing` needs no network and no weights; it exists so the
test suite runs anywhere, and it is never used for reported numbers.

---

## Reproducing every number

Run in order. Each stage caches, so a re-run is fast.

**1. Embeddings** (~4 min cold, instant cached):

```bash
python -m scripts.build_embeddings
```

**2. Train the kernel and evaluate both gates** (~3 min) — writes
`results/kernel_training.json`:

```bash
python -m scripts.train_kernel
```

**3. The experiment of record** (~35 min) — the ablation grid over all 300 SciFact
test queries, paired bootstrap against the baseline, quantum accounting, and the
poisoned-corpus arm. Writes `results/experiment.json`, checkpointing after every
system so an interrupted run still leaves usable results:

```bash
python -m scripts.run_experiment
```

A 40-query smoke test with three systems and no poisoning:

```bash
python -m scripts.run_experiment --quick --no-poison
```

**4. Tests** (~7 s):

```bash
python -m pytest tests -q
```

**5. Security audit** (~5 s):

```bash
python -m scripts.security_audit
```

Every result file embeds the config hash and the seed that produced it, so a table
in the report can be traced back to the exact configuration. All seeds are fixed at
`20260720`. Same command + same seed → same numbers.

### Optional: the demo API

```bash
uvicorn qrag.serve:app --host 127.0.0.1 --port 8000
```

Off by default and not required for any result. It exists so the supplied pre-deploy
checklist could be satisfied by a service that actually implements security headers,
rate limiting, restricted CORS and correlation-id error handling — rather than by a
report asserting that it would. See [SECURITY.md](SECURITY.md).

---

## How the honesty constraints are enforced in code

These are the parts most likely to be quietly violated under deadline pressure, so
each has a mechanism rather than an intention:

- **No test asserts a research result.** There is no `assert mrr > 0.7` anywhere.
  Such a test turns a measurement into a requirement and creates pressure to tune
  until it passes. Metrics live in `results/*.json` where they can be reproduced and
  disputed. The test suite's own docstring says this.
- **The quantum fast paths are cross-validated against qiskit-aer.** A numpy
  shortcut that is "algebraically equivalent" is an assumption until it is checked
  numerically.
- **The block kernel's non-equivalence to cosine is a test.** If it ever passes as
  rank-equivalent, the whole Q-RAG comparison is vacuous, and the test says so.
- **Complexity and simulation cost are separate columns.** Grover's oracle-query
  reduction is printed adjacent to its simulation overhead so the distinction cannot
  be lost in transcription into the report.
- **The exact optimum is computed alongside QAOA** at `n ≤ 16`, so the approximation
  quality is measured against a true optimum rather than against another heuristic.
- **The adversarial attack's weakness is stated wherever its numbers appear.** The
  embedding-optimised family is black-box (forward passes only) and therefore
  strictly weaker than the gradient-based attack of Zhong et al. (2023).
- **No silent caps.** If a run bounds coverage, it logs what was dropped.

---

## Layout

```
qrag/
  config.py        dataclass config, hashed into every result file
  data.py          BEIR loader (SciFact)
  embed.py         Ollama / sbert / hashing backends, content-addressed cache
  index.py         DenseIndex (faiss) + BM25Index
  kernel.py        GlobalFidelityKernel, BlockFidelityKernel, InfoNCE training
  fusion.py        per-query min-max normalisation and weighted fusion
  qsim.py          numpy statevector, cross-checked against qiskit-aer
  grover.py        amplitude amplification with a threshold oracle
  qaoa.py          selection QUBO, QAOA, exact brute force, affine-invariant quality
  pipeline.py      BaselineRAG and QRAG with independently switchable stages
  metrics.py       recall / precision / MRR / nDCG, paired bootstrap
  adversarial.py   four attack families, poisoning, attack reporting
  security.py      validation, resource ceilings, log redaction, injection detection
  serve.py         hardened demo API
scripts/
  build_embeddings.py  cache corpus + query embeddings
  train_kernel.py      fit both kernels, evaluate Gate A and Gate B
  run_experiment.py    the experiment of record
  security_audit.py    16 checks + 8 declared N/A, exits non-zero on failure
tests/
  test_qrag.py     68 tests: quantum correctness, kernel algebra, security controls
results/           measured numbers, tracked in git so tables can be checked
```

`cache/` (25 MB) and `data/` (11 MB) are gitignored and regenerated by step 1.

---

## Dataset

[BEIR SciFact](https://github.com/beir-cellar/beir) — 5,183 scientific abstracts,
300 test queries, 339 relevance judgments. Public research data under its own
licence. Downloaded on first use.

SciFact has a strong lexical baseline — a tuned hybrid reaches **recall@10 = 0.841
and nDCG@10 = 0.715** here — which leaves little headroom. That makes it a hard place
to show an improvement and an honest place to look for one.
