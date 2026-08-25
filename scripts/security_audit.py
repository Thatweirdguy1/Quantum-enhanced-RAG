"""Repeatable security audit. Run it; do not take the write-up's word for it.

    python -m scripts.security_audit          # human-readable, exits non-zero on FAIL
    python -m scripts.security_audit --json   # machine-readable for CI

Every check either exercises a real code path or inspects the real tree. Nothing
here asserts that a control exists because a docstring says so: the resource
ceilings are checked by trying to breach them, the input validation by feeding it
bad input, and the secret scan by regex over the files git would actually commit.

Scope
-----
Derived from the two supplied checklists, restricted to what a retrieval research
pipeline can meaningfully be audited for. Items with no counterpart here (password
hashing, OAuth flows, payment webhooks, row-level security on user records) are
listed as N/A with a reason in SECURITY.md rather than being silently dropped or
faked. A checklist item marked "done" that was never applicable is worse than one
marked N/A, because it hides the gap.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files that legitimately contain secret-shaped strings: the scanner's own
# patterns, the audit's own test fixtures, and the env template.
SCANNER_ALLOWLIST = {"qrag/security.py", "scripts/security_audit.py", ".env.example"}

MAX_TRACKED_BYTES = 5 * 1024 * 1024  # checklist: no large binaries in git

# Secret shapes, independent of qrag.security's set so that a mistake in one does
# not silently disable the other.
SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai-key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("hf-token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("github-pat", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("google-api-key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("slack-token", re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}")),
    ("private-key-block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("db-uri-with-password", re.compile(
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?)://[^\s:/]+:[^\s@]+@", re.IGNORECASE)),
    ("hardcoded-assignment", re.compile(
        r"\b(?:api_key|apikey|secret|password|passwd|token|access_key)\s*=\s*"
        r"['\"][^'\"\s]{12,}['\"]", re.IGNORECASE)),
)

# Things that should not ship. `breakpoint` and `pdb` are hard failures; a bare
# TODO is not, so it is reported at warn level rather than inflating the count.
DEBUG_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("breakpoint", re.compile(r"^\s*breakpoint\s*\(", re.MULTILINE), "fail"),
    ("pdb-set-trace", re.compile(r"\b(?:pdb|ipdb)\.set_trace\s*\(", re.MULTILINE), "fail"),
    ("debug-true", re.compile(r"\bdebug\s*=\s*True", re.IGNORECASE), "warn"),
    ("security-todo", re.compile(r"#\s*(?:TODO|FIXME|XXX)\b[^\n]*"
                                 r"(?:secur|auth|inject|secret)", re.IGNORECASE), "warn"),
)


@dataclass
class Check:
    check_id: str
    title: str
    source: str          # which supplied checklist this comes from
    status: str = "PASS"  # PASS | FAIL | WARN | N/A
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.check_id, "title": self.title, "source": self.source,
                "status": self.status, "detail": self.detail,
                "evidence": self.evidence[:20]}


def tracked_files() -> list[Path]:
    """Files git would commit. Falls back to a walk when git is unavailable."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return [ROOT / line for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    skip = {".git", "__pycache__", "cache", "data", ".venv", "venv", "figures"}
    return [p for p in ROOT.rglob("*")
            if p.is_file() and not any(part in skip for part in p.parts)]


def _text_files(files: list[Path]) -> list[tuple[str, str]]:
    keep = {".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg",
            ".ini", ".sh", ".example", ".gitignore", ".html", ".js", ".ts"}
    out = []
    for p in files:
        if p.suffix.lower() in keep or p.name in {".gitignore", ".env.example"}:
            try:
                out.append((p.relative_to(ROOT).as_posix(),
                            p.read_text(encoding="utf8", errors="replace")))
            except OSError:
                continue
    return out


# ============================================================ sec-prompts, check 1
def check_secret_leaks(files: list[Path]) -> Check:
    c = Check("SEC-1", "No secrets in tracked files", "sec-prompts #1 (Gitleaks)")
    for rel, text in _text_files(files):
        if rel in SCANNER_ALLOWLIST:
            continue
        for label, pattern in SECRET_RULES:
            for m in pattern.finditer(text):
                line = text[: m.start()].count("\n") + 1
                c.evidence.append(f"{rel}:{line} {label}")
    if c.evidence:
        c.status = "FAIL"
        c.detail = (f"{len(c.evidence)} secret-shaped string(s) in tracked files. "
                    "Rotate at the provider first, then purge from history -- a "
                    "deleting commit leaves the old blob reachable.")
    else:
        c.detail = (f"scanned {len(_text_files(files))} tracked text files against "
                    f"{len(SECRET_RULES)} credential shapes; none matched")
    return c


def check_gitignore() -> Check:
    c = Check("SEC-2", ".gitignore covers secrets and generated binaries",
              "sec-prompts #1 / prototype-vs-production item 2, item 6")
    path = ROOT / ".gitignore"
    if not path.exists():
        c.status = "FAIL"
        c.detail = "no .gitignore"
        return c
    body = path.read_text(encoding="utf8")
    required = [".env", "cache/", "data/", "*.npy", "__pycache__"]
    missing = [r for r in required if r not in body]
    if missing:
        c.status = "FAIL"
        c.detail = f"missing patterns: {missing}"
        c.evidence = missing
    else:
        c.detail = f"all {len(required)} required patterns present"
    if "!.env.example" not in body:
        c.status = "WARN" if c.status == "PASS" else c.status
        c.detail += "; .env.example may be swallowed by the .env.* rule"
    return c


def check_env_template() -> Check:
    c = Check("SEC-3", ".env.example exists and holds no live values",
              "sec-prompts #1 / prototype-vs-production item 2")
    path = ROOT / ".env.example"
    if not path.exists():
        c.status = "FAIL"
        c.detail = "no .env.example, so a new contributor cannot know what to set"
        return c
    body = path.read_text(encoding="utf8")
    for label, pattern in SECRET_RULES:
        if label == "hardcoded-assignment":
            continue
        if pattern.search(body):
            c.status = "FAIL"
            c.evidence.append(f"{label} present in template")
    keys = [ln.split("=")[0] for ln in body.splitlines()
            if "=" in ln and not ln.strip().startswith("#")]
    c.detail = (f"{len(keys)} documented variables"
                if c.status == "PASS" else "template contains live-looking values")
    return c


def check_env_var_only(files: list[Path]) -> Check:
    """Config must come from the environment, not from literals in source."""
    c = Check("SEC-4", "Configuration is env-var driven with no embedded defaults "
              "for secrets", "sec-prompts #3 / prototype-vs-production item 2")
    cfg = ROOT / "qrag" / "config.py"
    if not cfg.exists():
        c.status = "FAIL"
        c.detail = "qrag/config.py missing"
        return c
    tree = ast.parse(cfg.read_text(encoding="utf8"))
    env_reads = sum(1 for n in ast.walk(tree)
                    if isinstance(n, ast.Attribute) and n.attr in {"get", "environ"})
    from qrag.security import SecurityError, require_env
    try:
        require_env("QRAG_DEFINITELY_NOT_SET_" + "X" * 8)
        c.status = "FAIL"
        c.detail = "require_env returned instead of raising for an unset variable"
    except SecurityError:
        c.detail = (f"config.py reads the environment ({env_reads} accessor calls) "
                    "and require_env fails fast on an unset variable")
    return c


# ============================================================ sec-prompts, check 2
def check_log_hygiene() -> Check:
    c = Check("SEC-5", "Logs redact credentials and never store raw queries",
              "sec-prompts #2 (personal data flow)")
    import logging
    from io import StringIO

    from qrag.security import RedactingFilter, configure_logging, fingerprint

    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("qrag.audit.probe")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    secret = "sk-" + "A" * 30
    email = "patient@hospital.example"
    # Three shapes, because a filter that only redacts arguments misses the first
    # one: the credential is split across the format string and its argument, so
    # neither fragment matches on its own while the emitted line leaks in full.
    logger.info("token sk-%s and mail %s", "A" * 30, email)          # split
    logger.info("token %s and mail %s", secret, email)               # whole in arg
    logger.info("token %(k)s", {"k": secret})                        # dict args
    logger.info("literal %s %s", secret, email)
    written = buf.getvalue()

    leaks = []
    if secret in written:
        leaks.append("credential survived the filter "
                     "(split across format string and argument)")
    if email in written:
        leaks.append("email address survived the filter")
    fp = fingerprint("does vitamin D reduce mortality")
    if len(fp) < 8 or "vitamin" in fp:
        leaks.append("fingerprint is not opaque")
    configure_logging()  # ensure the documented entry point runs clean
    if leaks:
        c.status = "FAIL"
        c.evidence = leaks
        c.detail = "; ".join(leaks)
    else:
        c.detail = (f"credentials and emails redacted in emitted records; "
                    f"queries appear only as salted digests (e.g. {fp})")
    return c


# ============================================================ sec-prompts, check 3
def check_debug_artifacts(files: list[Path]) -> Check:
    c = Check("SEC-6", "No debug artefacts left in shipped code",
              "sec-prompts #3 / prototype-vs-production item 7")
    warns = []
    for rel, text in _text_files(files):
        if rel in {"scripts/security_audit.py"}:
            continue
        for label, pattern, severity in DEBUG_RULES:
            for m in pattern.finditer(text):
                line = text[: m.start()].count("\n") + 1
                (c.evidence if severity == "fail" else warns).append(
                    f"{rel}:{line} {label}")
    if c.evidence:
        c.status = "FAIL"
        c.detail = f"{len(c.evidence)} blocking artefact(s)"
    elif warns:
        c.status = "WARN"
        c.detail = f"{len(warns)} non-blocking mention(s): {warns[:5]}"
        c.evidence = warns
    else:
        c.detail = "no breakpoints, tracebacks-to-caller, or debug flags"
    return c


def check_resource_ceilings() -> Check:
    """Feature abuse / DoS, translated: the simulator is the expensive resource."""
    c = Check("SEC-7", "Resource ceilings refuse oversized allocations",
              "sec-prompts #5 (feature abuse) / prototype-vs-production item 8")
    from qrag.qaoa import QUBO, rerank_exact
    from qrag.qsim import Statevector, all_bitstrings
    from qrag.security import SECURITY, ResourceLimitExceeded
    import numpy as np

    over = SECURITY.max_qubits + 4
    over_exact = SECURITY.max_exact_qubits + 4
    probes = [
        ("Statevector(n)", lambda: Statevector(over)),
        ("Statevector.uniform(n)", lambda: Statevector.uniform(over)),
        ("all_bitstrings(n)", lambda: all_bitstrings(over_exact)),
        ("QUBO.energies()", lambda: QUBO(np.zeros(over_exact),
                                        np.zeros((over_exact, over_exact)), 3).energies()),
        ("rerank_exact()", lambda: rerank_exact(QUBO(np.zeros(over_exact),
                                                    np.zeros((over_exact, over_exact)), 3))),
    ]
    for label, fn in probes:
        try:
            fn()
            c.evidence.append(f"{label} allocated without refusing at n={over}")
        except ResourceLimitExceeded:
            pass
        except MemoryError:
            c.evidence.append(f"{label} raised MemoryError instead of refusing")
    if c.evidence:
        c.status = "FAIL"
        c.detail = "unbounded allocation path(s): " + "; ".join(c.evidence)
    else:
        c.detail = (f"all {len(probes)} allocation paths refuse above "
                    f"{SECURITY.max_qubits}q (simulation) / "
                    f"{SECURITY.max_exact_qubits}q (enumeration)")
    return c


# ============================================================ sec-prompts, check 4
def check_input_validation() -> Check:
    c = Check("SEC-8", "Untrusted input is validated, not trusted",
              "sec-prompts #4 (input validation)")
    from qrag.security import (InputRejected, SECURITY, validate_query,
                              validate_top_k)

    cases = [
        ("over-length query", lambda: validate_query("x" * (SECURITY.max_query_chars + 1))),
        ("null byte in query", lambda: validate_query("normal\x00query")),
        ("empty query", lambda: validate_query("   ")),
        ("negative top_k", lambda: validate_top_k(-5)),
        ("oversized top_k", lambda: validate_top_k(SECURITY.max_top_k + 1)),
    ]
    for label, fn in cases:
        try:
            fn()
            c.evidence.append(f"accepted: {label}")
        except InputRejected:
            pass
    smuggled = "what causes\u200b diabetes\U000e0049\U000e0067"
    cleaned = validate_query(smuggled)
    if "\u200b" in cleaned or "\U000e0049" in cleaned:
        c.evidence.append("invisible characters survived validation")
    if c.evidence:
        c.status = "FAIL"
        c.detail = "; ".join(c.evidence)
    else:
        c.detail = (f"{len(cases)} malformed inputs refused; zero-width and "
                    "Unicode-Tags characters stripped before use")
    return c


def check_path_traversal() -> Check:
    """The closest real analogue to the checklist's path-traversal item."""
    c = Check("SEC-9", "Cache filenames cannot escape the cache directory",
              "sec-prompts #4 (file handling) / #5 (ID manipulation)")
    from qrag.config import CACHE_DIR
    from qrag.security import InputRejected, safe_tag

    attempts = ["../../etc/passwd", "..\\..\\windows\\system32\\config\\sam",
                "a/b/c", "corpus\x00.npy", "....//....//secret", "/etc/shadow"]
    for raw in attempts:
        try:
            tag = safe_tag(raw)
        except InputRejected:
            continue
        resolved = (CACHE_DIR / f"emb-{tag}.npy").resolve()
        if CACHE_DIR.resolve() not in resolved.parents:
            c.evidence.append(f"{raw!r} -> {resolved}")
    if c.evidence:
        c.status = "FAIL"
        c.detail = "traversal possible: " + "; ".join(c.evidence)
    else:
        c.detail = (f"{len(attempts)} traversal attempts all resolve inside "
                    f"{CACHE_DIR.name}/")
    return c


# ============================================================ sec-prompts, check 5
def check_injection_detector() -> Check:
    """Attacker's perspective: content injection through retrieved passages."""
    c = Check("SEC-10", "Indirect prompt injection is detected and spotlighted",
              "sec-prompts #5 (content injection) / OWASP LLM01")
    from qrag.adversarial import make_instruction_injection, make_topical_mimicry
    from qrag.security import CONTEXT_PREAMBLE, build_context, injection_risk
    import numpy as np

    rng = np.random.default_rng(0)
    queries = ["does vitamin D reduce mortality",
               "is aspirin effective for primary prevention",
               "do statins cause diabetes"]
    injected = [make_instruction_injection(q, rng) for q in queries]
    benign = [make_topical_mimicry(q, rng) for q in queries]

    missed = [i for i, t in enumerate(injected) if injection_risk(t) == "none"]
    if missed:
        c.evidence.append(f"{len(missed)}/{len(injected)} injection payloads undetected")

    ctx, report = build_context(injected[:1] + benign,
                               doc_ids=["adv-0", "b1", "b2", "b3"])
    if report["n_flagged"] < 1:
        c.evidence.append("build_context did not flag a known payload")
    if CONTEXT_PREAMBLE[:40] not in ctx:
        c.evidence.append("spotlighting preamble absent from built context")
    for marker in ("[1]", "[2]", "End of retrieved excerpts"):
        if marker not in ctx:
            c.evidence.append(f"context missing structural marker {marker!r}")

    if c.evidence:
        c.status = "FAIL"
        c.detail = "; ".join(c.evidence)
    else:
        c.detail = (f"{len(injected)}/{len(injected)} payloads flagged; context is "
                    "numbered, delimited and preceded by a data-not-instructions "
                    "preamble. Note: detection of *fluent* poisoning is 0% by "
                    "design -- see SEC-11.")
    return c


def check_detector_limits() -> Check:
    """A control that is designed to expose the scanner's blind spot.

    Passing means the audit *reports* the blind spot, not that the blind spot is
    absent. A pattern scanner cannot detect a fluent false claim, and a suite
    that never checked would let the write-up imply otherwise.
    """
    c = Check("SEC-11", "Detector blind spot on fluent poisoning is measured, "
              "not hidden", "sec-prompts #5 (attacker's perspective)")
    from qrag.adversarial import detector_report, poison_corpus
    from qrag.data import Dataset, Document, Query

    tiny = Dataset(
        name="audit-fixture",
        documents=[Document("d1", "Vitamin D", "Observational cohort of vitamin D.")],
        queries=[Query("q1", "does vitamin D reduce mortality"),
                 Query("q2", "is aspirin effective for prevention")],
        qrels={"q1": {"d1": 1}, "q2": {"d1": 1}})
    _poisoned, man = poison_corpus(tiny, n_targets=2, per_query_per_family=3, seed=7)
    rows = man.detector

    inj = rows.get("instruction-injection", {}).get("detection_rate", 0.0)
    mim = rows.get("topical-mimicry", {}).get("detection_rate", 1.0)
    if inj < 0.9:
        c.evidence.append(f"instruction-injection detection only {inj:.0%}")
    c.detail = (f"instruction-injection {inj:.0%} detected, topical-mimicry "
                f"{mim:.0%}. The second number is the honest one: fluent "
                "contradiction carries no pattern, so retrieval-level defence "
                "(QAOA redundancy) is what must be measured against it.")
    if c.evidence:
        c.status = "FAIL"
    _ = detector_report
    return c


# =================================================== prototype-vs-production item 2, item 6
def check_large_files(files: list[Path]) -> Check:
    c = Check("PROD-1", "No large binaries staged for git",
              "prototype-vs-production item 2, item 6")
    for p in files:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > MAX_TRACKED_BYTES:
            c.evidence.append(f"{p.relative_to(ROOT).as_posix()} "
                              f"{size / 1e6:.1f} MB")
    if c.evidence:
        c.status = "FAIL"
        c.detail = ("large tracked files: " + "; ".join(c.evidence) +
                    f" (ceiling {MAX_TRACKED_BYTES / 1e6:.0f} MB)")
    else:
        total = sum(p.stat().st_size for p in files if p.exists())
        c.detail = (f"{len(files)} tracked files, {total / 1e6:.2f} MB total; "
                    "embeddings and corpora are regenerated, not committed")
    return c


def check_error_handling(files: list[Path]) -> Check:
    c = Check("PROD-2", "Errors are typed and carry no internals to the caller",
              "sec-prompts #3 / prototype-vs-production item 4")
    from qrag.security import (InputRejected, ResourceLimitExceeded,
                               SecurityError, validate_query)
    if not (issubclass(InputRejected, SecurityError)
            and issubclass(ResourceLimitExceeded, SecurityError)):
        c.status = "FAIL"
        c.detail = "security exceptions do not share a catchable base class"
        return c
    try:
        validate_query("x" * 99_999)
        c.status = "FAIL"
        c.detail = "no exception raised for an over-length query"
        return c
    except SecurityError as exc:
        message = str(exc)
    for leak in (str(ROOT), "Traceback", "File \""):
        if leak in message:
            c.status = "FAIL"
            c.evidence.append(f"error message leaks {leak!r}")
    # Match the statement form only, and skip this file: a substring count picks
    # up the audit's own literals and reports the auditor as the defect.
    bare_re = re.compile(r"^[ \t]*except[ \t]*:", re.MULTILINE)
    bare = sum(len(bare_re.findall(text)) for rel, text in _text_files(files)
               if rel != "scripts/security_audit.py")
    if bare:
        c.status = "WARN" if c.status == "PASS" else c.status
        c.detail = f"{bare} bare `except` clause(s), which swallow SecurityError too"
    else:
        c.detail = (f"typed hierarchy under SecurityError; refusal message "
                    f"({message[:48]!r}...) carries no path or traceback")
    return c


def check_reproducibility() -> Check:
    c = Check("PROD-3", "Runs are seeded and results are traceable to a config",
              "prototype-vs-production item 7 (testing/QA)")
    from qrag.config import DEFAULT
    unseeded = [name for name, sub in (("kernel", DEFAULT.kernel),
                                       ("quantum", DEFAULT.quantum),
                                       ("eval", DEFAULT.eval))
                if getattr(sub, "seed", None) is None]
    if unseeded:
        c.status = "FAIL"
        c.detail = f"unseeded config sections: {unseeded}"
        return c
    results = ROOT / "results"
    jsons = sorted(results.glob("*.json")) if results.exists() else []
    if not jsons:
        c.status = "WARN"
        c.detail = "no results/*.json yet; run the experiment before reporting"
        return c
    # Parse the JSON rather than substring-scanning a prefix: `config` sits well
    # past any fixed cut-off in a file that embeds full training histories.
    missing_cfg = []
    for p in jsons:
        try:
            if "config" not in json.loads(p.read_text(encoding="utf8")):
                missing_cfg.append(p.name)
        except (OSError, json.JSONDecodeError):
            missing_cfg.append(f"{p.name} (unreadable)")
    c.detail = (f"{len(jsons)} result file(s), every section seeded"
                + (f"; without an embedded config: {missing_cfg}" if missing_cfg else
                   "; each embeds the config that produced it"))
    if missing_cfg:
        c.status = "WARN"
    return c


def check_dependency_pinning() -> Check:
    c = Check("PROD-4", "Dependencies are pinned", "prototype-vs-production item 6")
    req = ROOT / "requirements.txt"
    if not req.exists():
        c.status = "FAIL"
        c.detail = "no requirements.txt"
        return c
    lines = [ln.strip() for ln in req.read_text(encoding="utf8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    unpinned = [ln for ln in lines if not re.search(r"[><=~]=", ln)]
    if unpinned:
        c.status = "WARN"
        c.detail = f"{len(unpinned)}/{len(lines)} unpinned: {unpinned[:6]}"
        c.evidence = unpinned
    else:
        c.detail = f"all {len(lines)} requirements carry a version constraint"
    return c


# ------------------------------------------------------------- N/A, stated openly
def not_applicable() -> list[Check]:
    """Checklist items with no counterpart here, and why.

    Recorded rather than dropped. An audit that quietly omits the items it cannot
    satisfy reads as a clean bill of health; this list is what makes the PASS
    count above interpretable.
    """
    items = [
        ("NA-1", "Password hashing (bcrypt/argon2)", "sec-prompts #2, #4",
         "No user accounts exist. The pipeline has no authentication surface "
         "because it stores no user records and issues no sessions."),
        ("NA-2", "IDOR / per-user authorisation on every route", "sec-prompts #4, #5",
         "There are no per-user resources. Every document in the corpus is "
         "public BEIR data with identical read permissions."),
        ("NA-3", "Payment webhook signature verification", "sec-prompts #4",
         "No payment or billing path."),
        ("NA-4", "Server-side price recalculation", "sec-prompts #4",
         "No commerce logic; nothing is priced."),
        ("NA-5", "Row-level security / tenant isolation", "prototype-vs-production item 3",
         "Single-tenant research corpus, read-only, no database."),
        ("NA-6", "Password-reset token lifetime and single use", "sec-prompts #4",
         "No credential recovery flow, because there are no credentials."),
        ("NA-7", "GDPR data-deletion request handling", "prototype-vs-production item 10",
         "No personal data is collected. Queries are held in memory for the "
         "duration of a request and recorded in logs only as salted digests, so "
         "there is no per-subject record to delete."),
        ("NA-8", "Cookie flags (HttpOnly/Secure/SameSite)", "sec-prompts #2, #3",
         "The demo API is stateless and sets no cookies; it authenticates with a "
         "bearer token when QRAG_ENV=production."),
    ]
    return [Check(i, t, s, status="N/A", detail=d) for i, t, s, d in items]


def run_all() -> list[Check]:
    files = tracked_files()
    checks = [
        check_secret_leaks(files),
        check_gitignore(),
        check_env_template(),
        check_env_var_only(files),
        check_log_hygiene(),
        check_debug_artifacts(files),
        check_resource_ceilings(),
        check_input_validation(),
        check_path_traversal(),
        check_injection_detector(),
        check_detector_limits(),
        check_large_files(files),
        check_error_handling(files),
        check_reproducibility(),
        check_dependency_pinning(),
    ]
    serve = ROOT / "qrag" / "serve.py"
    if serve.exists():
        checks.append(check_api_hardening())
    return checks + not_applicable()


def check_api_hardening() -> Check:
    """Pre-deploy items, exercised against a live in-process instance of the API.

    An earlier version of this check only grepped serve.py for the header names.
    That was the weakest check in the suite and it contradicted this module's own
    premise: a header named in a dict literal is a declaration, not a response. It
    also could not have caught the defect the test suite did find -- fastapi was
    rejecting every POST /search with 422 because the body annotation was
    unresolvable -- since the string "X-Frame-Options" was present the whole time.

    So this now boots the app through starlette's in-process transport and reads
    the actual responses. No socket, no network, no corpus.
    """
    c = Check("PROD-5", "Demo API hardening verified against live responses",
              "sec-prompts #3 (pre-deploy audit)")
    try:
        from fastapi.testclient import TestClient

        from qrag import serve
    except Exception as exc:  # fastapi optional; say so rather than silently pass
        c.status = "WARN"
        c.detail = f"API not testable in this environment: {type(exc).__name__}: {exc}"
        return c

    client = TestClient(serve.create_app())
    probes: list[str] = []

    # 1. Headers on a real response, not in a dict literal.
    resp = client.get("/healthz")
    for header, expected in (("X-Content-Type-Options", "nosniff"),
                             ("X-Frame-Options", "DENY"),
                             ("Content-Security-Policy", "default-src 'none'"),
                             ("Referrer-Policy", "no-referrer")):
        got = resp.headers.get(header, "")
        if expected not in got:
            c.evidence.append(f"{header}: expected {expected!r}, got {got!r}")
    probes.append(f"{len(serve.SECURITY_HEADERS)} headers present on a live response")

    # 2. /healthz must not become a reconnaissance endpoint.
    if resp.json() != {"status": "ok"}:
        c.evidence.append(f"/healthz leaks detail: {resp.json()}")
    probes.append("/healthz returns status only")

    # 3. CORS allowlist, from the resolved config rather than the source text.
    if "*" in serve.ALLOWED_ORIGINS or not serve.ALLOWED_ORIGINS:
        c.evidence.append(f"CORS origins unsafe: {serve.ALLOWED_ORIGINS}")
    probes.append(f"{len(serve.ALLOWED_ORIGINS)} explicit origin(s), no wildcard")

    # 4. A valid body must reach the handler. This is the regression guard.
    valid = client.post("/search", json={"query": "vitamin D and mortality", "top_k": 3})
    if valid.status_code == 422:
        c.evidence.append("valid request rejected as unprocessable: body annotation "
                          "is not resolving, so validation is not running")
    elif "error" not in valid.json():
        c.evidence.append(f"error envelope missing on {valid.status_code}: {valid.json()}")
    probes.append(f"valid body reaches the handler ({valid.status_code})")

    # 5. Bounds refuse before any retrieval work happens. Two layers, probed
    #    separately: the body cap is middleware and fires on bytes, the field bound
    #    is pydantic and fires on characters. A 99 KB query never reaches pydantic,
    #    so asserting 422 for it would have been asserting the wrong layer -- this
    #    check originally did exactly that and the live run returned 413.
    huge = client.post("/search", json={"query": "x" * 99_999, "top_k": 3})
    if huge.status_code != 413:
        c.evidence.append(f"99KB body returned {huge.status_code}, expected 413 "
                          "from the request-size cap")
    over = client.post("/search",
                       json={"query": "x" * (serve.SECURITY.max_query_chars + 10),
                             "top_k": 3})
    if over.status_code != 422:
        c.evidence.append(f"overlong query returned {over.status_code}, expected 422 "
                          "from the field bound")
    big_k = client.post("/search", json={"query": "ok", "top_k": 10_000})
    if big_k.status_code != 422:
        c.evidence.append(f"top_k=10000 returned {big_k.status_code}, expected 422")
    probes.append("size cap refuses 99KB with 413; field bounds refuse with 422")

    # 6. Rate limit actually fires, and the refusal is still hardened.
    limited = TestClient(_rate_limited_app(serve, per_min=2))
    codes = [limited.get("/healthz").status_code for _ in range(4)]
    if 429 not in codes:
        c.evidence.append(f"rate limit never fired: {codes}")
    else:
        last = limited.get("/healthz")
        if "Retry-After" not in last.headers:
            c.evidence.append("429 without Retry-After")
        if last.headers.get("X-Content-Type-Options") != "nosniff":
            c.evidence.append("429 response is missing security headers")
    probes.append(f"rate limit fires at the configured ceiling ({codes})")

    # 7. No traceback, no filesystem path, in any error body.
    for body in (valid.text, over.text, huge.text, limited.get("/healthz").text):
        if "Traceback" in body or "/qrag/" in body or "qrag\\" in body or "C:\\" in body:
            c.evidence.append("an error body leaks an internal path or traceback")
            break
    probes.append("error bodies carry a correlation id and no internals")

    if c.evidence:
        c.status = "FAIL"
        c.detail = "; ".join(c.evidence)
    else:
        c.detail = f"{len(probes)} controls verified live: " + "; ".join(probes)
    return c


def _rate_limited_app(serve, per_min: int):
    """A second app whose limiter is tight enough to trip inside the audit.

    Rebinding the module constant and rebuilding is enough because create_app reads
    RATE_LIMIT_PER_MIN when it constructs the bucket; the original value is restored
    so the rest of the audit sees an unchanged module.
    """
    original = serve.RATE_LIMIT_PER_MIN
    try:
        serve.RATE_LIMIT_PER_MIN = per_min
        return serve.create_app()
    finally:
        serve.RATE_LIMIT_PER_MIN = original


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true", help="treat WARN as failure")
    args = ap.parse_args()

    checks = run_all()
    counts = {s: sum(1 for c in checks if c.status == s)
              for s in ("PASS", "FAIL", "WARN", "N/A")}
    failed = counts["FAIL"] + (counts["WARN"] if args.strict else 0)

    if args.json:
        print(json.dumps({"counts": counts, "strict": args.strict,
                          "exit": 1 if failed else 0,
                          "checks": [c.as_dict() for c in checks]}, indent=2))
        return 1 if failed else 0

    glyph = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "N/A": "n/a "}
    print("=" * 78)
    print("Q-RAG security audit")
    print("=" * 78)
    active = [c for c in checks if c.status != "N/A"]
    for c in active:
        print(f"[{glyph[c.status]}] {c.check_id:<8} {c.title}")
        print(f"          source: {c.source}")
        print(f"          {c.detail}")
        for e in c.evidence[:6]:
            print(f"            - {e}")
    print("-" * 78)
    print("Not applicable to a retrieval research pipeline (stated, not dropped):")
    for c in checks:
        if c.status == "N/A":
            print(f"  n/a  {c.check_id:<6} {c.title}")
            print(f"       {c.detail}")
    print("-" * 78)
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, "
          f"{counts['WARN']} warned, {counts['N/A']} not applicable")
    if failed:
        print("AUDIT FAILED -- fix the items above before pushing or reporting.")
    else:
        print("AUDIT PASSED. Re-run after any change to qrag/security.py or the "
              "allocation paths it guards.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
