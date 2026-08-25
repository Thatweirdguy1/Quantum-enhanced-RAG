# Security audit

Two checklists were supplied for this project:

- **`sec-prompts.pdf`** — five security checks before launch (Gitleaks, Bearer,
  ECC Production Audit, Trail of Bits, ECC Security Review).
- **`Prototype_vs_Production_Checklist.pdf`** — ten sections on the gap between a
  working prototype and a deployable system.

Both are written for a web application with user accounts, payments and a database.
Q-RAG is a retrieval research pipeline: a corpus of public scientific abstracts, an
embedding model, a quantum-simulated reranker, and no users. Applying the checklists
honestly therefore meant doing three separate things, and the distinction matters
more than the pass rate:

1. **Translate** the items that have a real analogue. SQL injection has no meaning
   here, but *indirect prompt injection through retrieved passages* is the same
   class of bug — untrusted data reaching an interpreter — and it is the single most
   relevant threat to a RAG system. That translation is done explicitly, not
   silently.
2. **Implement** the items that apply as written: secrets, log hygiene, input
   validation, resource ceilings, error handling, rate limiting, CORS, headers.
3. **Declare N/A with a reason** for the items that genuinely do not exist. Eight
   items are marked not-applicable below. None of them are marked N/A because they
   were hard.

Everything claimed here is checked by a script, not asserted in prose:

```bash
python -m scripts.security_audit
```

Current status: **16 passed, 0 failed, 0 warned, 8 not applicable.** The audit exits
non-zero on any failure, so it can gate a commit.

```bash
python -m pytest tests -q
```

**68 passed.** Roughly half of those tests attack the controls below rather than
exercise the happy path — a control that has never refused anything is untested code.

---

## The three defects this process found in my own code

This is the part of an audit that is worth reading. A checklist that finds nothing
has usually been run as a formality.

### 1. Request validation on the demo API was dead code

`POST /search` returned HTTP 422 for **every** request, including valid ones. The
cause was a subtle interaction: `qrag/serve.py` uses `from __future__ import
annotations`, so every annotation is a string, and FastAPI resolves annotations with
`typing.get_type_hints`, which searches **module globals only**. `SearchRequest` was
defined inside `create_app()`, so it was unresolvable; FastAPI fell back to treating
the request body as a *query* parameter, and the body bounds — `max_length`,
`ge`/`le` on `top_k` — never ran.

- **How an attacker exploits it:** they do not need to. The endpoint was closed to
  everyone. But had the fallback gone the other way, the field bounds that stop an
  oversized query from reaching the pipeline would have been absent while appearing
  present in the source.
- **Fix:** the models and the FastAPI imports are now at module scope, with a
  comment recording why they cannot be moved back — [qrag/serve.py:112](qrag/serve.py#L112).
- **What caught it:** `tests/test_qrag.py`. The audit's original check for this item
  *grepped `serve.py` for the header names*, so it passed throughout — the string
  `X-Frame-Options` was in the file the whole time. A header named in a dict literal
  is a declaration, not a response.
- **Consequence for the audit:** check PROD-5 was rewritten to boot the app through
  Starlette's in-process transport and read the actual responses. On its first live
  run it immediately found the 413/422 layering discrepancy in item 3 below. There
  is now a regression test whose whole purpose is to assert that a valid body
  reaches the handler.

### 2. A credential could be logged in full while passing the redaction filter

The redacting log filter matched against `record.msg`. But `logger.info("token
sk-%s", key)` splits the secret across `record.msg` and `record.args`, so neither
fragment matched the credential pattern — while the *emitted line* contained the
whole thing.

- **How an attacker exploits it:** reads it out of the log file, or out of any
  log-aggregation service the file is shipped to. Log files routinely have weaker
  access control than the secret store they defeat.
- **Fix:** the filter now redacts `record.getMessage()` — the fully interpolated
  string — and clears `record.args`, so nothing can be reassembled downstream.
- **Test:** `test_split_secret_across_msg_and_args_is_still_redacted`.

### 3. Two unbounded allocations behind a ceiling that could not fire

`qrag/security.py` enforces a qubit ceiling, because a statevector is `2**n`
complex128 and an unvalidated qubit count is a one-line memory exhaustion. Two paths
bypassed it:

- `Statevector.uniform(n)` called `np.full(2**n, ...)` — the allocation happened
  while evaluating the *argument*, before the ceiling check in the body could run.
- `all_bitstrings(n)` and the qiskit-aer path had no ceiling at all.

- **How an attacker exploits it:** any request that reaches a qubit count from input
  crashes the process. `n=40` asks for 8 TiB.
- **Fix:** the ceiling is checked before allocation on all five paths, with separate
  limits for simulation (22 qubits) and exhaustive enumeration (20).
- **Test:** `test_resource_ceilings_refuse_oversized_allocations` covers all five.

### 4. A reported quantum metric was silently inverted

Not a security defect, but found the same way and worth recording next to the
others: the QAOA reranker reported `approximation_ratio = achieved / optimal`. The
QUBO reduces to `−Σrᵢ + λΣ Sᵢⱼ` on feasible selections, which is negative when
relevance dominates but **positive** when a query's candidates are weakly relevant
and mutually similar. In the positive regime that ratio rewards being *worse*: the
first full run reported `1.0058`, which read literally claims QAOA beat a
brute-force optimum over the same feasible set — impossible. It meant QAOA was 0.58%
**worse**. Replaced with an affine-invariant quality in `[0, 1]`
(`(worst − achieved) / (worst − optimal)`), which is immune to the sign, plus an
explicit `is_optimal` flag and the gap in objective units — [qrag/qaoa.py](qrag/qaoa.py).
The existing test compared raw objectives and so could not have caught it; there is
now one that constructs the positive regime deliberately.

---

## `sec-prompts.pdf` — the five checks

### Check 1 — Secret leak prevention (Gitleaks)

| Item | Status |
|---|---|
| All secrets in environment variables | **Done.** No credential literal exists in the source tree. Config is read through `qrag.config` and `qrag.security.require_env` / `optional_env` (10 accessor calls). |
| Supabase / Stripe / DB URI / OAuth / JWT keys | **N/A** — none of these services are used. There is no database, no payment processor and no OAuth provider. |
| Frontend exposure (`NEXT_PUBLIC_`, `REACT_APP_`) | **N/A** — there is no frontend bundle. The demo API serves JSON only, and its CSP is `default-src 'none'`. |
| `.env` in `.gitignore`, `.env.example` committed | **Done.** `.gitignore` ignores `.env` and `.env.*` with an explicit `!.env.example` negation. The template documents 17 variables with placeholder values. |
| Logs and responses do not leak secrets | **Done**, after fixing defect 2 above. |
| Git history warning | **Done** — stated in [README.md](README.md) and in `.env.example`. Deleting a secret in a new commit does **not** remove it; the old blob is still reachable, so rotate at the provider and rewrite with `git filter-repo` or BFG. |

**Audit checks:** SEC-1 (34 tracked text files scanned against 10 credential
shapes), SEC-2, SEC-3, SEC-4.

The one secret this project actually has is `QRAG_API_TOKEN`, the demo API's shared
bearer token. When `QRAG_ENV=production` it is required **with no default** and the
process refuses to boot without it. A service that silently starts unauthenticated
is worse than one that does not start.

### Check 2 — Personal data flow audit (Bearer)

The data map is short, which is the point.

| Data | Where it enters | Where it goes | Retention |
|---|---|---|---|
| Search query text | `POST /search`, or a script argument | The retrieval pipeline, in memory. Sent to the local Ollama daemon on `127.0.0.1` for embedding. | Duration of the request. Never written to disk. |
| Query *fingerprint* | Derived at log time | Log line only | As long as logs are kept |
| Client IP | Connection metadata | Rate-limit bucket (in memory, 60 s window); logged **as a salted digest**, never raw | 60 s in the bucket |
| Corpus documents | BEIR SciFact release, public | Index, on disk under `data/` | Persistent, and public data |

- **Logs.** Queries appear only as salted digests — `fingerprint(query)` — so a log
  file cannot be mined for what was searched. The salt (`QRAG_LOG_SALT`) is what
  stops an attacker precomputing digests of common queries; rotating it
  deliberately breaks correlation with old logs. A `logging.Filter` redacts
  credential shapes and email addresses from every record.
- **Third-party integrations.** One: the Ollama daemon, over loopback by default. It
  receives query text and passage text, because that is what an embedding model is
  for, and nothing else — no identifiers, no IP, no headers. No analytics, no error
  tracking, no telemetry of any kind is installed.
- **Passwords, cookies, `localStorage`, API response filtering, data deletion** —
  **N/A** (audit items NA-1, NA-7, NA-8). No accounts, no cookies, no browser
  storage, no per-subject record to delete. The `/search` response returns doc ids,
  scores, titles and an injection-risk label; it has no user fields to over-return.

**Audit check:** SEC-5.

### Check 3 — Pre-deploy production audit (ECC)

Every item verified against **live in-process responses**, not by reading the source
(see defect 1 for why that distinction is load-bearing).

| Item | Implementation |
|---|---|
| Env vars referenced properly, refuse to start if critical ones missing | `require_env` raises with a hint naming the variable; `QRAG_API_TOKEN` is required in production |
| Debug code removed, debug mode off by default | SEC-6: no breakpoints, no tracebacks to the caller, no debug flags. `/docs` and `/openapi.json` are disabled when `QRAG_ENV=production` |
| No stack traces to the client | Every handler returns `{"error": "...", "correlation_id": "..."}`. The traceback goes to the log with the same id, so an operator can find it and the client cannot see it. A single `HTTPException` handler guarantees one envelope for the whole API — otherwise a 401 would carry no correlation id and a failed auth attempt could not be tied to a log line |
| Security headers on every response | Six, set by middleware with `setdefault`: `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Cross-Origin-Opener-Policy`, `Permissions-Policy`, and `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'`. HSTS (`max-age=63072000`) is sent over TLS or in production only — sending it on plain HTTP in development would pin `localhost` to https in the developer's browser and be a nuisance to undo |
| Rate limiting | Fixed-window per-IP limiter, `QRAG_RATE_LIMIT_PER_MIN` (default 20), verified firing at the configured ceiling. **Honest scope:** it is in-process, so with N workers the effective limit is N × the setting. A real deployment needs a shared store or a limit at the reverse proxy. This is stated in the code, not glossed |
| CORS not `*` | Explicit allowlist; a literal `*` is filtered out of the env value, and `allow_credentials=False`. The combination the checklist singles out is unreachable from this configuration |
| Database TLS, no default credentials | **N/A** — no database |

The checklist asks for 5 attempts/minute on login and 3/hour on password reset.
There is no login and no password reset, so those specific numbers have nothing to
apply to; the generic per-IP limit covers the endpoints that do exist.

**Audit checks:** PROD-2, PROD-5 (7 controls verified live), SEC-6.

### Check 4 — Deep audit of complex logic (Trail of Bits)

*The app has:* a read-only retrieval pipeline over a public corpus, a
quantum-simulated reranker, and a single-token demo API. No payments, no user auth,
no smart contracts.

- **Authentication & authorisation** — one bearer token, compared with
  `hmac.compare_digest` so the comparison is constant-time. IDOR, ownership checks,
  password reset and JWT handling are **N/A** (NA-2, NA-6): there are no per-user
  resources. Every document is public BEIR data with identical read permissions.
- **Payment logic** — **N/A** (NA-3, NA-4). Nothing is priced.
- **Input handling** — this is where the translation happens, and it is the
  substantial part of this audit:

| Checklist item | Q-RAG analogue | Control |
|---|---|---|
| SQL injection | **Indirect prompt injection** via retrieved passages (OWASP LLM01) — untrusted text reaching an interpreter, which here is the LLM | `injection_risk()` classifies five families; `build_context()` numbers and delimits every passage behind a data-not-instructions preamble, and defangs fences and role markers rather than deleting content |
| XSS | Unicode smuggling in the query | `validate_query` applies NFKC normalisation and strips the **Tags block U+E0000–U+E007F** (invisible ASCII smuggling), zero-width characters and bidi overrides |
| File upload validation | Cache filename construction | Every cache path is built from a sanitised slug plus a content hash and asserted to resolve inside `CACHE_DIR`; 6 traversal payloads tested |
| Raw SQL → parameterised | No SQL exists | — |

**Audit checks:** SEC-8, SEC-9, SEC-10.

One deliberate design decision: `validate_query` **refuses** an oversized query
rather than truncating it. Truncation is a silent partial execution of an
attacker-influenced input; a refusal is a decision the caller can see. There is a
test asserting the refusal specifically, because a future "helpful" truncation would
otherwise pass unnoticed.

### Check 5 — Attacker's perspective (ECC Security Review)

| Attack path | Finding |
|---|---|
| ID manipulation to reach another user's data | **N/A** — no per-user data (NA-2) |
| Login bypass, expired/malformed tokens, default admin | No default credentials exist. In production the token is mandatory; a malformed or absent `Authorization` header yields 401 with a correlation id |
| Privilege escalation | No roles |
| **Feature abuse / DoS** | **Real and applicable.** The quantum simulator is the attack surface: `2ⁿ` allocation from an input-derived qubit count. Ceilings enforced on all five allocation paths *before* allocation (defect 3). Body size capped at 16 KB by middleware, so a 99 KB body is refused on bytes and never reaches pydantic — cheapest layer first, and the two layers are tested separately |
| Content injection | See check 4. Also measured, not just implemented — see below |
| **Internal exposure** | `/healthz` returns `{"status": "ok"}` and nothing else. A readiness probe that reports which model is loaded is reconnaissance. OpenAPI/Swagger disabled in production. `.env` gitignored, `.git` not served (no static file handler exists) |
| Business logic manipulation | No commerce logic |

**Audit checks:** SEC-7, SEC-8, SEC-9, SEC-10, SEC-11.

#### The measured blind spot

This is the finding I would want a reader to take seriously. The pattern-based
injection detector is **100% effective against instruction injection and 0%
effective against topical mimicry** — and the second number is the honest one.

Fluent, well-formed text that simply contradicts the scientific claim carries no
detectable pattern. There is nothing to match on: it looks exactly like a real
abstract, because that is what it was written to look like. Any claim that a regex
layer "defends against corpus poisoning" is false, and it is false in a way that is
easy to hide behind a high aggregate detection rate.

So the poisoned-corpus arm of the experiment exists to test whether a
**retrieval-level** defence works where the pattern layer cannot: the QAOA
redundancy penalty `λΣ_{i<j} sᵢⱼxᵢxⱼ` should suppress *clusters* of mutually-similar
injected passages regardless of how fluent they are, because it penalises the
similarity between selected documents rather than their content. The control that
makes this falsifiable is `qrag[no-qaoa]` — without it, any drop in context
occupancy could be credited to the kernel instead. Results and the verdict are in
[results/experiment.json](results/experiment.json); the numbers are reported whether
or not the hypothesis held.

Four adversarial families are injected (`qrag/adversarial.py`): topical mimicry,
lexical/BM25 gaming, black-box embedding optimisation, and instruction injection.
The embedding-optimised attack uses **forward passes only** — no gradients — and is
therefore **strictly weaker** than the gradient-based attack of Zhong et al. (2023).
That caveat is stated wherever the numbers appear, because reporting a defence
against a weak attack as a defence in general is the standard way this kind of
result gets overstated. Query relevance judgments (`qrels`) are never modified, so
retrieval metrics stay comparable across the clean and poisoned arms; a test asserts
that.

---

## `Prototype_vs_Production_Checklist.pdf` — the ten sections

| # | Section | Status |
|---|---|---|
| 1 | Authentication & access control | Bearer token, constant-time compare, mandatory in production. No accounts by design (NA-1, NA-2, NA-6) |
| 2 | Secrets & environment variables | SEC-1 – SEC-4. `.env.example` with 17 documented variables; rotation warning in README |
| 3 | Data & database | No database (NA-5). Corpus is public, read-only. Data flow mapped under check 2 |
| 4 | Error handling & screen states | PROD-2. Typed hierarchy under `SecurityError`; generic client messages + correlation id; refusal messages carry a limit, never a path or traceback |
| 5 | Hosting & deployment | Local/single-node scope, stated as such. No cloud deployment is claimed |
| 6 | Version control & code quality | PROD-1: 31 tracked files, 1.56 MB. Embeddings (25 MB) and corpora (11 MB) are gitignored and regenerated, not committed. PROD-4: all 17 requirements carry a version constraint |
| 7 | Testing & QA | 68 tests. PROD-3: every run seeded (`seed=20260720`) and every result file traceable — an experiment result embeds the config hash that reproduces it, a verification artefact embeds the commit it ran against, and a file carrying neither is flagged |
| 8 | Performance & optimisation | SEC-7 resource ceilings. Latency is reported honestly: the statevector simulator is **30.7× slower** than the classical baseline and no wall-clock win is claimed. Grover is reported as an **oracle-query** result with simulation overhead in an adjacent column |
| 9 | Maintenance & monitoring | Structured logging with correlation ids; `/healthz`; the audit script is re-runnable as a gate |
| 10 | Legal & compliance | BEIR SciFact is public research data under its own licence. No personal data is collected (NA-7), so there is no GDPR subject-access or deletion obligation to implement |

---

## The eight not-applicable items, and why

Stated rather than dropped, because "not applicable" is only credible when it is
specific:

| id | Item | Why it does not apply |
|---|---|---|
| NA-1 | Password hashing (bcrypt/argon2) | No user accounts. No authentication surface, because nothing stores user records or issues sessions |
| NA-2 | IDOR / per-user authorisation | No per-user resources. Every corpus document is public with identical read permissions |
| NA-3 | Payment webhook signature verification | No payment or billing path |
| NA-4 | Server-side price recalculation | No commerce logic |
| NA-5 | Row-level security / tenant isolation | Single-tenant read-only research corpus, no database |
| NA-6 | Password-reset token lifetime and single use | No credential recovery flow, because there are no credentials |
| NA-7 | GDPR data-deletion handling | No personal data collected. Queries live in memory for one request and appear in logs only as salted digests, so there is no per-subject record to delete |
| NA-8 | Cookie flags (HttpOnly/Secure/SameSite) | Stateless API, sets no cookies; authenticates with a bearer token |

---

## What this audit does not cover

Both source documents say some version of this, and it applies here:

- **No human penetration test.** Everything above is self-audit plus automated
  checks. The controls have been attacked by my own tests, which share my own blind
  spots.
- **The demo API is a demo.** It exists so that the pre-deploy items could be
  satisfied by a service that actually implements them rather than by a report
  claiming it would. It is not intended for public exposure, its rate limiter is
  per-worker, and it has no TLS termination of its own.
- **The injection detector is a defence in depth layer, not a solution.** Its blind
  spot is measured above and is large.
- **Ollama is trusted.** Query and passage text go to a local daemon in cleartext
  over loopback. If `OLLAMA_URL` is pointed at a remote host, that traffic leaves the
  machine unencrypted and the data-flow map above stops being accurate.
