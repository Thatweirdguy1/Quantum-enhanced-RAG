r"""Run the test suite and the security audit, and record what they returned.

    python -m scripts.verify

Writes ``results/verification.json``.

Why this exists: the report, the slide deck and the diary all state how many tests
pass and how many audit checks pass. Typing those counts into a document copies them
out of a terminal scrollback, and a scrollback goes stale the moment a test is added
or a check starts failing -- at which point the document asserts a pass that no
longer happens. So the counts are parsed from the harnesses themselves and read from
this file by every document that quotes them.

Neither harness is timed, so this is safe to run at any point, unlike
``scripts.run_experiment``, whose latency figures are only valid from an
uninterrupted run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from qrag.provenance import snapshot as provenance_snapshot

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "verification.json"

# pytest's terminal summary, e.g. "68 passed in 6.91s" or "1 failed, 67 passed in 7s".
_COUNT = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")


def run_pytest() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()
    summary = tail[-1] if tail else ""
    counts = {kind: int(n) for n, kind in _COUNT.findall(summary)}
    return {
        "command": "python -m pytest tests -q",
        "exit": proc.returncode,
        "summary": summary,
        "counts": counts,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0) + counts.get("error", 0)
                  + counts.get("errors", 0),
    }


def run_audit() -> dict:
    """Run the security audit and read back its counts.

    One quirk worth knowing: the audit inspects ``results/*.json``, which includes
    this script's own outputs from the *previous* invocation. So a change to what
    those files contain takes two runs to settle -- the first run is still scanning
    the old generation. If a count looks wrong immediately after changing a result
    writer, run it again before believing it.
    """
    audit_out = ROOT / "results" / "audit.json"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.security_audit", "--out", str(audit_out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    counts = {}
    if audit_out.exists():
        counts = json.loads(audit_out.read_text(encoding="utf8"))["counts"]
    return {
        "command": "python -m scripts.security_audit",
        "exit": proc.returncode,
        "counts": counts,
        "passed": counts.get("PASS", 0),
        "failed": counts.get("FAIL", 0),
        "warned": counts.get("WARN", 0),
        "not_applicable": counts.get("N/A", 0),
    }


def main() -> int:
    tests = run_pytest()
    audit = run_audit()
    payload = {
        "provenance": provenance_snapshot(),
        "tests": tests,
        "audit": audit,
        # A document may only claim "verified" if both harnesses came back clean.
        # Deriving that here means no document has to make the judgement itself.
        "all_clean": tests["exit"] == 0 and audit["exit"] == 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf8")

    print(f"tests: {tests['summary'] or '(no summary line)'}  -> exit {tests['exit']}")
    print(f"audit: {audit['passed']} passed, {audit['failed']} failed, "
          f"{audit['warned']} warned, {audit['not_applicable']} not applicable"
          f"  -> exit {audit['exit']}")
    print(f"wrote {OUT.relative_to(ROOT)}  (all_clean={payload['all_clean']})")
    return 0 if payload["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
