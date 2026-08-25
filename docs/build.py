r"""Build the report deliverables from their Markdown sources.

    python -m docs.build lit          # literature review
    python -m docs.build paper        # research paper
    python -m docs.build diary        # daily diary
    python -m docs.build all

Each target is a Markdown file in ``docs/`` rendered to ``build/`` by
:mod:`docs.amity_docx`. The prose lives in the Markdown so it can be read, diffed
and edited without touching code, and the formatting lives in one renderer so the
three documents cannot drift into three different house styles.

Page counts are estimated rather than measured: python-docx writes the file but does
not lay it out, so the true count depends on the renderer that opens it. The
estimate uses the text area of a Letter page at 12 pt with 1.5 spacing and is
accurate to roughly a page over a document of this length -- enough to tell whether
a 20-page requirement is met, not enough to be quoted as exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

from docs.amity_docx import build
from docs.facts import Facts, FactError

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build"

TARGETS = {
    "lit": ("literature_review.md", "Q-RAG_Literature_Review.docx"),
    "paper": ("research_paper.md", "Q-RAG_Research_Paper.docx"),
    "diary": ("daily_diary.md", "Q-RAG_Daily_Diary.docx"),
}

# 6.5 x 9 inch text area, 12 pt at 1.5 spacing is 18 pt per line, so 36 lines per
# page; justified Times New Roman averages close to 13 words per line.
WORDS_PER_PAGE = 36 * 13


def estimate_pages(markdown: str) -> tuple[int, int]:
    """Rough (words, pages) for a Markdown source.

    Table rows are counted for their text as well as their row overhead. An earlier
    version skipped ``|`` lines entirely, which reported a table-heavy document as
    three pages when nearly all of its prose lived in table cells.
    """
    words = 0
    rows = 0
    for line in markdown.splitlines():
        s = line.strip()
        if not s or s.startswith(("[[", "---", "```")):
            continue
        if s.startswith("|"):
            # A separator row (|---|---|) carries no text.
            if set(s) <= set("|-: "):
                continue
            rows += 1
            words += len(s.replace("|", " ").split())
            continue
        words += len(s.split())
    # A row costs at least a line even when its cells are short, and headings plus
    # their space-before cost roughly a line each.
    lines_from_text = words / 13
    return words, max(1, round((lines_from_text + rows * 0.4) / 36 + 0.5))


def main(argv: list[str]) -> int:
    which = argv[1] if len(argv) > 1 else "all"
    names = list(TARGETS) if which == "all" else [which]
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        print(f"unknown target(s): {', '.join(unknown)}\n"
              f"available: {', '.join(TARGETS)}, all")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    facts = Facts()
    missing = []
    for name in names:
        src_name, out_name = TARGETS[name]
        src = ROOT / "docs" / src_name
        if not src.exists():
            missing.append(f"{name}: {src.relative_to(ROOT)} not written yet")
            continue

        raw = src.read_text(encoding="utf8")
        # Resolve {{token}} placeholders against results/*.json before rendering. A
        # bad token raises here rather than producing a document with a visible
        # placeholder or a blank where a measured number belongs.
        try:
            text = facts.substitute(raw)
        except FactError as exc:
            print(f"{name:<6} FAILED -- {exc}")
            missing.append(f"{name}: unresolved fact tokens (see above)")
            continue

        n_tokens = raw.count("{{")
        resolved = ROOT / "build" / f"_resolved_{src_name}"
        resolved.write_text(text, encoding="utf8")
        target = build(resolved, OUT / out_name)
        resolved.unlink()

        words, pages = estimate_pages(text)
        print(f"{name:<6} -> {target.relative_to(ROOT)}  "
              f"({words:,} words, ~{pages} pages, "
              f"{target.stat().st_size / 1024:.0f} KB, "
              f"{n_tokens} facts substituted)")
    for line in missing:
        print(line)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
