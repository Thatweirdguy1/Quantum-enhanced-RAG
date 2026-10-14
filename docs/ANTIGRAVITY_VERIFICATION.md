# Q-RAG Independent Verification Report (Pre-Submission Audit)

**Date of Verification:** 15 October 2026, 02:00 IST
**Evaluator:** Antigravity (Independent Model Cowork Integrator & Verification Agent)
**Target Repository:** `c:\Users\prabh\Downloads\QERAG`
**Git HEAD Tested:** `6899d36` (*Align SRS and paper with PolyForm Noncommercial license*)
**Submission Archive Tested:** `build/Q-RAG_Submission_2026-10-15.zip` (32,303,921 bytes / ~30.8 MB)
**Overall Verdict:** **Automated checks and live SciFact workbench smoke test pass; human layout review remains open.**

---

## 1. Executive Summary & Verification Matrix

The tested archive, automated checks, and CI passed. A subsequent live Streamlit smoke test with the installed Ollama models also passed. These checks do not establish that the documents look correct to a human reader.

| Verification Item | Target Standard | Observed Result | Verdict |
|---|---|---|---|
| **Cold-Start Test Suite** | 80 tests passing, 0 failures | **80 passed**, 1 warning in 16.00s (local), 17.13s (isolated ZIP) | **PASS** |
| **Security Audit** | 16 pass, 0 fail, 0 warn, 8 N/A | **16 PASS**, 0 FAIL, 0 WARN, 8 N/A (exit 0) | **PASS** |
| **Submission Archive (ZIP)** | Fresh build, testable source | Clean extraction, all 80 tests pass inside fresh temp environment; Ollama weights are not bundled | **PASS for tests** |
| **GitHub Actions CI (`ci.yml`)** | Commit `6899d36` clean | Run #8: `completed`, conclusion: `success` (Ubuntu & Windows) | **PASS** |
| **GitHub CodeQL Analysis** | Extended scanning clean | Run #26: `completed`, conclusion: `success` (0 open alerts) | **PASS** |
| **License Alignment (Tracked)** | PolyForm Noncommercial 1.0.0 | Verified in `LICENSE`, `README.md`, `docs/research_paper.md`, `deliverables/` | **PASS** |
| **Streamlit Modes & Features** | Workbench, Ingestion, Fallbacks | Live SciFact Full Q-RAG retrieval, three-mode comparison, generation, separate-model judge, and ranked CSV export passed; upload path was not exercised | **PARTIAL** |
| **Deliverables & PDF Previews** | Document integrity & page counts | PDF page counts: Paper 27, Lit Review 39, Diary 9; human visual review not recorded | **PARTIAL** |
| **Stitch Mockup Licensing** | Must not claim MIT if published | Line 495 of untracked `about_methodology/code.html` claims MIT | **DEFECT (Low)** |

---

## 2. Cold-Start Reproduction Steps

The following sequence was verified from a clean shell in `c:\Users\prabh\Downloads\QERAG`:

### Step 1: Environment & Runtime Verification
```powershell
python --version
# Output: Python 3.13.15
```

### Step 2: Test Suite Execution
```powershell
.venv\Scripts\activate
python -m pytest tests -q
```
**Result summary:**
```
........................................................................ [ 90%]
........                                                                 [100%]
80 passed, 1 warning in 16.00s
```
*Note on warning:* `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` is an upstream Starlette/FastAPI library deprecation and does not affect test correctness.

### Step 3: Security & Production Audit
```powershell
python -m scripts.security_audit
```
**Exact Output:**
```
======================================================================
Q-RAG security audit
======================================================================
...
16 passed, 0 failed, 0 warned, 8 not applicable
AUDIT PASSED
```

### Step 4: Isolated Submission ZIP Execution
Extracted `build/Q-RAG_Submission_2026-10-15.zip` to a clean temporary directory (`%TEMP%\qrag_verify_*`) and executed `pytest`:
```powershell
python -m pytest tests -q
# Output: 80 passed, 1 warning in 17.13s (exit code 0)
```

---

## 3. GitHub CI & CodeQL Confirmation (Resolved Risk)

Prior handoffs noted that GitHub CI for commit `6899d36` was unconfirmed. Direct inspection via the GitHub REST API (`https://api.github.com/repos/Thatweirdguy1/Quantum-enhanced-RAG/actions/runs`) confirms both workflows completed with **success**:

1. **Continuous Integration (`.github/workflows/ci.yml`)**:
   - Run ID: `38629257125` (Run #8)
   - Commit SHA: `6899d36c996659ee20f61ada1c184945e35e8f78`
   - Trigger: Push to `main`
   - Matrix: `ubuntu-latest`, `windows-latest`
   - Status: `completed`
   - Conclusion: `success`
   - Completed at: `2026-10-14T20:08:21Z`

2. **CodeQL Code Scanning**:
   - Run ID: `38629257350` (Run #26)
   - Commit SHA: `6899d36c996659ee20f61ada1c184945e35e8f78`
   - Status: `completed`
   - Conclusion: `success`
   - Completed at: `2026-10-14T20:07:35Z`

---

## 4. Comprehensive License Wording Audit

The following tracked sources and submission artifacts were checked for license consistency; the untracked Stitch mockup remains inconsistent:

| Location | Expected License | Found License | Status |
|---|---|---|---|
| `LICENSE` | PolyForm Noncommercial 1.0.0 | Full PolyForm text with Copyright 2026 @thatweirdguy1 (Prabhav Goel) | **Consistent** |
| `README.md` (Line 6) | PolyForm Noncommercial | Explicit link to `[PolyForm Noncommercial License](LICENSE)` | **Consistent** |
| `docs/research_paper.md` (L669) | PolyForm Noncommercial | *"The software source is publicly available under the PolyForm Noncommercial License 1.0.0. It is not offered under MIT..."* | **Consistent** |
| `deliverables/Q-RAG_Research_Paper.docx` | PolyForm Noncommercial | Matches paper source text; MIT disclaimed | **Consistent** |
| `deliverables/QRAG_SRS_IEEE_Amity_Logo.docx` | PolyForm Noncommercial | *"This is an academic research project. The software source is publicly available on GitHub under the PolyForm Noncommercial License 1.0.0..."* | **Consistent** |
| `build/Q-RAG_Submission_2026-10-15.zip` | PolyForm Noncommercial | Contains root `LICENSE` and updated DOCX deliverables | **Consistent** |
| `stitch_q_rag_research_workbench/.../about_methodology/code.html` (L495) | PolyForm Noncommercial | `Code, statevector simulation logs, and adversarial probe datasets are archived under open MIT and CC-BY-4.0 licensing.` | ⚠️ **DEFECT D1** |

---

## 5. Research Workbench (Streamlit) & Fallback Verification

Static inspection of `qrag/workbench.py` and pipeline paths found the following capabilities. This is not evidence of a successful live UI session:

1. **Interface Modes**:
   - `Workbench`: Live query execution against SciFact or dynamic user uploads, candidate inspection, retrieval latency metric, grounded generation, and independent judge verification.
   - `Compare modes`: Parallel execution across Classical Hybrid, Kernel-only, and Full Q-RAG pipelines on identical query vectors.
   - `Evaluation`: Reads `results/experiment.json`, renders nDCG@10 comparison bar charts, and displays bootstrap 95% confidence intervals with explicit non-significance disclaimers.
   - `Security`: Visualizes the 4-family poisoned corpus attack, context occupancy metrics, and pattern detector blind spots.
   - `Methodology`: Explicitly declares simulation on classical hardware and disclaims wall-clock speedup.
2. **File Ingestion**:
   - Accepts multiple TXT or PDF documents up to 20 files. Chunks into overlapping segments via `qrag.data.ingest_documents`, computes dense embeddings on the fly, and builds ephemeral FAISS + BM25 indexes.
3. **Robust Fallbacks**:
   - Small corpus (< 4 candidates): Grover amplification is automatically bypassed (`result.trace["grover"]["skipped"] = "fewer than four candidates"`).
   - QAOA optimizer non-convergence / budget exhaustion: Falls back safely to greedy top-k with audit trace logging (`result.trace["qaoa"]["fallback"] = "greedy top-k"`).
4. **CSV Export Security**:
   - Single-query hit exports sanitize cells against formula injection by prefixing `'` to values beginning with `=`, `+`, `-`, or `@`.
   - Per-query recorded experiment CSV (`results/experiment.csv`, 3,302 lines, 2.82 MB) is directly downloadable from the UI.

---

## 6. Document Deliverables & Previews Inspection

Generated deliverables were checked for text and structure. PDF page counts are exact for the current previews; Word layout and logo alignment still need a human visual review:

- **Research Paper**: `deliverables/Q-RAG_Research_Paper.docx` (PDF preview: 27 pages). Includes Amity formatting, IEEE citation style, abstract, empirical tables matching `FACTS.md`, and explicit disclaimers of speedup.
- **Literature Review**: `deliverables/Q-RAG_Literature_Review.docx` (39 pages). Substantially exceeds the departmental 20-page requirement, covering quantum retrieval theoretical limits, NISQ simulation costs, and dense-lexical hybrid baselines.
- **Daily Diary**: `deliverables/Q-RAG_Daily_Diary.docx` (PDF preview: 9 pages). Records chronological project development, kernel derivation, CodeQL remediations, and verification runs.
- **Viva Presentation**: `deliverables/Q-RAG_Slides.pptx` (10 slides). Concise summary of problem statement, architecture, negative retrieval result, Grover oracle complexity, QAOA security finding, and honest viva talking points.
- **Software Requirements Specification (SRS)**: `deliverables/QRAG_SRS_IEEE_Amity_Logo.docx` (~39.9 KB). Complete IEEE 830 format aligned with PolyForm Noncommercial License 1.0.0.

---

## 7. Subsequent live workbench smoke test (15 October 2026)

Streamlit launched on `127.0.0.1:8501` with installed `bge-m3:latest`, `qwen3.5:latest`, and `deepseek-r1:8b` Ollama models. In the browser, the SciFact question “Does vitamin D supplementation reduce mortality?” returned five ranked passages in Full Q-RAG mode (9,532.0 ms retrieval time, 98 candidates). The Compare modes action completed and displayed its results table. With “Generate and judge answer” enabled, the same question produced a cited answer and an independent judge result (relevance 0.90, grounded: yes). Generation took substantially longer than retrieval. The user subsequently supplied the manually downloaded `qrag_retrieval.csv`; it contains the expected six columns and five ranked rows, with nonempty passages and the same five titles and scores shown in the UI. TXT/PDF upload and its fallback behavior were not exercised in this smoke test.

## 8. Prioritized Defect Register

| Defect ID | Severity | Location | Description | Recommended Remediation |
|---|---|---|---|---|
| **D1** | **Low** | `stitch_q_rag_research_workbench/stitch_q_rag_research_workbench/about_methodology/code.html:495` | Untracked Stitch HTML mockup states: *"Code, statevector simulation logs, and adversarial probe datasets are archived under open MIT and CC-BY-4.0 licensing."* | Correct the software-license claim before tracking or sharing the mockup; verify any separate dataset or content license before retaining a CC-BY-4.0 claim. |
| **D2** | **Informational** | `results/verification.json` | Records `git_commit: "f1ec08d"`, which is 3 commits behind current `HEAD` (`6899d36`). | Run `python -m scripts.verify` if an updated automated checkpoint artifact is desired. |
| **D3** | **Informational** | `build/_paper_resolved.txt` | Stale intermediate text dump from 26 August 2026 reflecting an earlier draft with 68 tests. | Exclude this obsolete, ignored file from submission; the current build command uses and removes a differently named temporary file. |
| **D4** | **Informational** | Root working directory | Stray untracked backtick file `` ` `` present. | Remove or add to `.gitignore`. |

---

## 9. Final Recommendation

The automated test, security, CI, archive, live SciFact workbench, and ranked CSV export checks pass. Before calling the complete project submission-ready, inspect the final document pages, TOCs, and logos visually; an upload/fallback check is also advisable. Do not include the untracked Stitch mockup without correcting its license claim.
