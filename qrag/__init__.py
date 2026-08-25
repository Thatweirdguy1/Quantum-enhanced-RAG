"""Q-RAG: Quantum-Enhanced Retrieval-Augmented Generation.

A retrieval-augmented generation system that layers three quantum-inspired
components over a classical RAG baseline and benchmarks against it:

* ``qrag.kernel``  -- phase-modulated fidelity kernel generalising cosine
                      similarity, with trainable phase parameters.
* ``qrag.grover``  -- amplitude amplification over a pre-filtered shortlist.
* ``qrag.qaoa``    -- top-k reranking framed as a QUBO and solved by QAOA.

The retriever/generator boundary is deliberately explicit: ``BaselineRAG`` and
``QRAG`` in :mod:`qrag.pipeline` implement the same ``Retriever`` protocol, so
switching between them is a configuration change rather than a code path.
"""

__version__ = "0.6.0"  # week 6 checkpoint

GROUP = "165"
INSTITUTION = "Amity School of Engineering & Technology"
TEAM = (
    ("A2305223498", "Prabhav Goel"),
    ("A2305223509", "Saksham Singh"),
    ("A2305223505", "Sameen Ur Rehman"),
)
