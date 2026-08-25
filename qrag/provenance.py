r"""Where a result file came from.

Every file written into ``results/`` records enough to identify the code and machine
that produced it. This is a single shared helper rather than a copy per writer,
because ``security_audit.py`` check PROD-3 asserts that every result file is
traceable, and three near-identical local versions of the same dictionary is how one
of them quietly stops carrying the commit.

An experiment result adds the config hash and seed on top of this (see
``scripts/run_experiment.py``); a verification artefact has no retrieval config to
hash, so this snapshot is the whole of its traceability.
"""

from __future__ import annotations

import platform
import subprocess
import sys


def git_commit() -> str:
    """Short HEAD, or a stated reason it is unavailable.

    Never raises and never returns an empty string: a result file that silently
    omits the commit looks identical to one produced outside version control.
    """
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else "not-a-git-repo"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def snapshot() -> dict:
    """Code and platform identity for a result file."""
    out = {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import numpy as np
        out["numpy"] = np.__version__
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        out["numpy"] = "unavailable"
    return out
