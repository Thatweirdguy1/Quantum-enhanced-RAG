"""Local Streamlit interface for live retrieval and recorded experiments.

Run with ``streamlit run qrag/workbench.py`` from the repository root.
"""

from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import sys
import uuid

import pandas as pd
# Streamlit executes this file as a script, so make repository imports work even
# when invoked from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from qrag.config import RESULTS_DIR
from qrag.pipeline import BaselineRAG, QRAG
from qrag.runtime import load_runtime
from qrag.security import injection_risk, validate_query


st.set_page_config(page_title="Q-RAG Research Workbench", page_icon="Q",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
  :root { --line:#293247; --muted:#AEBBD0; --panel:#121826; }
  .stApp { background:#080B12; }
  [data-testid="stSidebar"] { background:#0D111B; border-right:1px solid var(--line); }
  h1,h2,h3 { letter-spacing:-.025em; }
  h1 { color:#F2F5FA; }
  p,li { line-height:1.55; }
  [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:16px; min-height:106px; }
  [data-testid="stMetricLabel"] { color:var(--muted); }
  [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:8px; }
  div.stButton > button[kind="primary"] { background:#8B7CF6; color:#080B12; border:0; }
  div.stButton > button { min-height:42px; }
  div[data-baseweb="tab-list"] { border-bottom:1px solid var(--line); gap:8px; }
  div[data-baseweb="tab"] { font-weight:600; }
  :focus-visible { outline:2px solid #8B7CF6!important; outline-offset:2px; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading SciFact, embeddings, and trained kernel…")
def runtime():
    return load_runtime()


@st.cache_data(show_spinner=False)
def experiment():
    path = RESULTS_DIR / "experiment.json"
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def pipeline_for(rt, mode: str):
    base = rt.pipeline
    if mode == "Classical":
        return BaselineRAG(rt.dataset, base.dense, base.bm25, {}, base.cfg)
    return QRAG(rt.dataset, base.dense, base.bm25, {}, base.kernel, base.cfg,
                use_kernel=True, use_interference=mode == "Full Q-RAG",
                use_grover=mode == "Full Q-RAG", use_qaoa=mode == "Full Q-RAG")


def run_query(rt, query: str, top_k: int, mode: str, answer: bool):
    query = validate_query(query)
    vector = rt.embedder.encode([query])[0]
    result = pipeline_for(rt, mode).retrieve(uuid.uuid4().hex, query,
                                             top_k=top_k, query_vector=vector)
    docs = [rt.dataset.doc(doc_id) for doc_id in result.ranked]
    generated = judged = None
    if answer:
        passages = [doc.content for doc in docs]
        ids = [doc.doc_id for doc in docs]
        generated = rt.generator.generate(query, passages, ids)
        judged = rt.judge.evaluate(query, generated.text, passages, ids)
    return result, docs, generated, judged


def hits_csv(result, docs) -> str:
    def cell(value):
        text = str(value)
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text

    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["rank", "doc_id", "title", "score", "injection_risk", "content"])
    for rank, doc in enumerate(docs, 1):
        writer.writerow([rank, cell(doc.doc_id), cell(doc.title),
                         result.scores.get(doc.doc_id, 0.0),
                         injection_risk(doc.content), cell(doc.content)])
    return out.getvalue()


def show_hits(result, docs, *, key: str):
    st.subheader("Retrieved context")
    st.caption("Scores are relative to this query and retrieval mode. Passage text is untrusted source content.")
    for rank, doc in enumerate(docs, 1):
        risk = injection_risk(doc.content)
        title = doc.title or doc.doc_id
        with st.expander(f"{rank:02d} · {title} · score {result.scores.get(doc.doc_id, 0):.3f}" +
                         (f" · {risk} injection risk" if risk != "none" else "")):
            st.caption(f"Document ID: {doc.doc_id} · Injection risk: {risk}")
            st.write(doc.content)
    st.download_button("Download results CSV", hits_csv(result, docs),
                       file_name="qrag_retrieval.csv", mime="text/csv", key=key)


st.sidebar.markdown("### Q-RAG")
st.sidebar.caption("Research workbench · local simulator")
mode = st.sidebar.radio("Retrieval mode", ["Classical", "Kernel", "Full Q-RAG"],
                        index=2, help="The full mode includes interference, Grover and QAOA.")
top_k = st.sidebar.slider("Results to retrieve", 1, 20, 5)
generate_answer = st.sidebar.checkbox("Generate and judge answer", value=False,
                                       help="Uses local Ollama generator and a separate judge model; may take minutes.")
st.sidebar.divider()
st.sidebar.caption("All quantum stages run on a classical simulator. Live query timing excludes model loading and answer generation.")

st.title("Q-RAG Research Workbench")
st.caption("SciFact retrieval · grounded answers · reproducible evaluation")
workbench, compare, evaluation, security, about = st.tabs(
    ["Workbench", "Compare modes", "Evaluation", "Security", "Methodology"])

with workbench:
    st.subheader("Ask the corpus")
    with st.form("search"):
        query = st.text_input("Research question", placeholder="Does vitamin D supplementation reduce mortality?",
                              max_chars=2000)
        submitted = st.form_submit_button("Run retrieval", type="primary")
    if submitted:
        try:
            with st.spinner("Searching the corpus and, if selected, generating an answer…"):
                st.session_state["run"] = (mode, query, *run_query(runtime(), query, top_k,
                                                                    mode, generate_answer))
        except Exception as exc:
            st.error(f"Run failed: {exc}")
    if "run" in st.session_state:
        run_mode, run_query_text, result, docs, generated, judged = st.session_state["run"]
        st.caption(f"Query: {run_query_text} · Mode: {run_mode}")
        a, b, c = st.columns(3)
        a.metric("Retrieval time", f"{result.latency_ms:.1f} ms")
        b.metric("Candidates", result.trace.get("pool_size", "—"))
        c.metric("Flagged passages", sum(injection_risk(d.content) != "none" for d in docs))
        if generated:
            st.subheader("Grounded answer")
            st.write(generated.text)
            if judged:
                st.info(f"Independent judge · relevance {judged.relevance:.2f} · "
                        f"grounded: {'yes' if judged.grounded else 'no'} · {judged.explanation}")
            with st.expander("Context safety report"):
                st.json(generated.context_report)
        if result.trace:
            with st.expander("Retrieval trace"):
                st.json(result.trace)
        show_hits(result, docs, key="workbench_csv")
    else:
        st.info("Enter a question and run retrieval to inspect real ranked passages.")

with compare:
    st.subheader("Compare retrieval modes")
    st.caption("Runs the same query vector through all three modes. Full Q-RAG can take about a second after model loading.")
    with st.form("comparison"):
        comparison_query = st.text_input("Comparison question", value=query or "Does vitamin D supplementation reduce mortality?")
        compare_now = st.form_submit_button("Compare modes")
    if compare_now:
        try:
            with st.spinner("Running three retrieval modes…"):
                rt = runtime()
                text = validate_query(comparison_query)
                vector = rt.embedder.encode([text])[0]
                st.session_state["comparison_results"] = {
                    name: pipeline_for(rt, name).retrieve(uuid.uuid4().hex, text,
                                                          top_k=top_k, query_vector=vector)
                    for name in ("Classical", "Kernel", "Full Q-RAG")}
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
    if "comparison_results" in st.session_state:
        results = st.session_state["comparison_results"]
        st.dataframe([{"Mode": name, "Retrieval ms": round(res.latency_ms, 1),
                       "Top document": res.ranked[0] if res.ranked else "—",
                       "Top 5 IDs": ", ".join(res.ranked[:5])}
                      for name, res in results.items()], hide_index=True, width="stretch")
        st.caption("These are individual query results; benchmark metrics below come from the full recorded experiment.")

data = experiment()
with evaluation:
    st.subheader("Recorded SciFact evaluation")
    if data is None:
        st.warning("results/experiment.json is missing. Run the experiment to populate this view.")
    else:
        st.caption(f"{data['dataset']['n_queries']} test queries · {data['dataset']['n_docs']:,} documents · "
                   f"seed {data['config']['eval']['seed']} · config {data['provenance']['config_hash']}")
        rows = []
        for name, entry in data["clean"].items():
            m = entry["metrics"]
            rows.append({"System": name, "nDCG@10": round(m["ndcg@10"], 4),
                         "Recall@10": round(m["recall@10"], 4),
                         "MRR@10": round(m["mrr@10"], 4),
                         "Mean latency ms": round(m["latency_ms_mean"], 1)})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.bar_chart(pd.DataFrame(rows).set_index("System")["nDCG@10"],
                     horizontal=True, x_label="nDCG@10")
        full = data["significance"]["qrag[full]"]["ndcg@10"]
        st.info(f"Full Q-RAG vs classical: ΔnDCG@10 {full['delta']:+.4f}; "
                f"95% CI [{full['ci95_low']:+.4f}, {full['ci95_high']:+.4f}]. "
                "The measured difference is not statistically significant.")
        st.download_button("Download benchmark CSV", "\n".join(
            ["system,ndcg_at_10,recall_at_10,mrr_at_10,mean_latency_ms"] +
            [f"{r['System']},{r['nDCG@10']},{r['Recall@10']},{r['MRR@10']},{r['Mean latency ms']}" for r in rows]),
            file_name="qrag_benchmark.csv", mime="text/csv")

with security:
    st.subheader("Adversarial corpus experiment")
    if data is None:
        st.warning("Recorded experiment data is unavailable.")
    else:
        manifest = data["poisoned"]["manifest"]
        st.caption(f"{manifest['n_injected']} injected passages across {manifest['n_target_queries']} target queries; "
                   "five context slots per query.")
        st.dataframe([{"System": name,
                       "Poisoned context occupancy": f"{entry['attack']['context_occupancy']:.1%}",
                       "Clean context rate": f"{entry['attack']['clean_context_rate']:.1%}",
                       "Top 10 poison hit rate": f"{entry['attack']['top_k_hit_rate']:.1%}"}
                      for name, entry in data["poisoned"]["systems"].items()],
                     hide_index=True, width="stretch")
        occupancy = data["poisoned"]["systems"]["qrag[full]"]["attack"]["context_occupancy"]
        st.warning(f"Full Q-RAG reduces poisoned context occupancy in this measured attack, "
                   f"but {occupancy:.1%} of its context slots remain poisoned. This is not a safety guarantee.")
        detector = manifest["detector"]
        st.dataframe([{"Attack family": name, "Flagged": f"{v['flagged']}/{v['n']}",
                       "Detection rate": f"{v['detection_rate']:.0%}"}
                      for name, v in detector.items()], hide_index=True,
                     width="stretch")
        st.caption("The rule detector misses all three non-instruction attack families in this test.")

with about:
    st.subheader("What this system measures")
    st.write("Q-RAG combines BM25 and dense retrieval with a trained phase kernel, "
             "interference reranking, Grover shortlist simulation, and QAOA context selection. "
             "The comparison uses the same SciFact queries and corpus across modes.")
    st.write("All quantum circuits are simulated on a classical computer. "
             "The experiment does not establish a wall-clock speedup or a retrieval-quality advantage.")
    st.markdown("[Read the methods and reproducible results on GitHub]"
                "(https://github.com/Thatweirdguy1/Quantum-enhanced-RAG)")
