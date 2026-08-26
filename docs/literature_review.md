---
title: Quantum-Enhanced Retrieval-Augmented Generation
subtitle: A Critical Literature Review
degree: Submitted in partial fulfilment of the requirements for the degree of Bachelor of Technology in Computer Science and Engineering
authors: Prabhav Goel; [ADD REMAINING GROUP 165 MEMBERS]
supervisor: [ADD SUPERVISOR NAME AND DESIGNATION]
department: Department of Computer Science and Engineering
institution: Amity School of Engineering and Technology
date: August 2026
footer: Q-RAG Literature Review
---

# Abstract

This review surveys the three bodies of work that a quantum-enhanced
retrieval-augmented generation system necessarily draws on: retrieval-augmented
generation itself, quantum and quantum-inspired information retrieval, and the
quantum search and optimisation subroutines that such a system might use as
components. It also surveys a fourth literature that work in this area routinely
omits, namely the security of retrieval-augmented systems against corpus poisoning
and indirect prompt injection.

The review is deliberately critical rather than encyclopaedic. Its organising
argument is that the quantum information retrieval literature contains a recurring
and under-acknowledged failure mode: a scoring function is introduced as quantum,
is shown to be expressible as a fidelity or inner-product overlap, and is then
found on inspection to be a monotone transformation of the classical similarity it
was meant to improve on. A monotone transformation cannot change a ranking. Where
that is true, the quantum component of the system is a no-op with respect to the
task, and any measured difference in retrieval quality must have come from
somewhere else. We trace this problem through the quantum language model and
complex-valued matching literatures, connect it to the formal results on quantum
kernels and their classical shadows, and identify the structural conditions under
which a fidelity-based score can escape it.

Two secondary conclusions follow from the survey. First, the two quantum
subroutines most often proposed for retrieval, Grover search and the Quantum
Approximate Optimization Algorithm, offer guarantees in units that are not
wall-clock time, and conflating the two is the second most common error in the
applied literature. Second, defences against corpus poisoning are frequently
evaluated against attacks weaker than the strongest published ones, and reported in
aggregate in a way that conceals which attack families they fail against entirely.

---

[[TOC]]

[[PAGEBREAK]]

# 1. Introduction

## 1.1 Motivation

Retrieval-augmented generation has become the default architecture for grounding a
large language model in a specific document collection. The pattern is simple
enough to state in a sentence: retrieve a small number of passages relevant to the
user's query, place them in the model's context, and generate an answer conditioned
on them. Its appeal is that it addresses the two most commercially damaging
failure modes of a bare language model, namely confident fabrication and a fixed
knowledge cut-off, without retraining the model.

The retrieval step is therefore load-bearing. If the passages placed in context are
wrong, the generated answer is wrong in a way that is harder to detect than an
ungrounded hallucination, because it arrives with an apparent citation. This has
made the ranking function at the heart of retrieval a subject of sustained
attention, and it is the point at which quantum computing has been proposed as an
intervention.

The proposal is not obviously unreasonable. Retrieval is, in the abstract, a search
problem over a large unstructured collection, and search is where the earliest and
best-known quantum speed-up applies. Similarity between a query and a document is
conventionally measured by an inner product between real-valued vectors, and quantum
mechanics offers a richer object in the same shape: an inner product between complex
amplitude vectors, in which relative phase carries information that a real inner
product cannot represent. Selecting a small, diverse subset of candidates to place
in a limited context window is a combinatorial optimisation problem of exactly the
form that variational quantum algorithms target.

Each of those three analogies has been pursued in the literature. This review
examines what each has actually delivered, and is unusually concerned with a
question that survey papers in this area tend to pass over: whether the quantum
component of a proposed system does anything at all.

## 1.2 Scope and selection criteria

Sources were selected on three criteria. First, peer-reviewed venues were preferred
over preprints, with preprints included where they are the canonical reference for a
widely used method, as is the case for the Quantum Approximate Optimization
Algorithm. Second, the review prefers work that reports a measured comparison
against a competent classical baseline over work that reports only a construction.
Third, where a claim is contested, both sides are cited; the dequantisation results
discussed in Section 5.4 are included precisely because they undercut earlier
speed-up claims that are still cited approvingly.

Two exclusions should be stated. The review does not cover quantum hardware
engineering, error correction, or the physical realisation of qubits, because no
part of the accompanying project runs on hardware. It also does not cover quantum
natural language processing in the compositional, categorical sense associated with
DisCoCat and its successors; that line of work addresses grammatical structure
rather than retrieval ranking, and including it would broaden the review without
informing the system under construction.

## 1.3 Organisation

Section 2 establishes the classical baseline that any quantum proposal must be
measured against, and is deliberately detailed, because the most common weakness in
applied quantum retrieval papers is a classical baseline that has not been tuned.
Section 3 covers the quantum computing background that bears on retrieval, with
particular attention to data encoding, which is where most proposed advantages are
lost. Section 4 surveys quantum and quantum-inspired information retrieval and
develops the review's central critical argument. Section 5 covers quantum kernel
methods, which supply the formal machinery for stating that argument precisely.
Section 6 covers Grover search and QAOA as subroutines. Section 7 covers the
security literature. Section 8 synthesises the gap that the accompanying project
addresses, and Section 9 concludes.

[[PAGEBREAK]]

# 2. Retrieval-Augmented Generation

## 2.1 From open-domain question answering to RAG

The architecture now called RAG emerged from open-domain question answering.
Chen et al. (2017) established the two-stage retriever-reader pattern, using a
sparse retriever to select Wikipedia articles and a neural reader to extract an
answer span. The retriever in that work was a fixed lexical function, and the
learning happened entirely in the reader.

Two 2020 papers moved the learning into the retriever and set the shape of the
field. Karpukhin et al. (2020) introduced Dense Passage Retrieval, training a
dual-encoder with a contrastive objective so that a query embedding and its relevant
passage embedding are close under an inner product, and demonstrated that a learned
dense retriever could outperform BM25 on open-domain QA. Lewis et al. (2020)
introduced the term retrieval-augmented generation for an architecture in which a
dense retriever is coupled to a sequence-to-sequence generator and the two are
trained jointly, with the retrieved document treated as a latent variable
marginalised over during generation. Guu et al. (2020) pursued a related idea in
REALM, pre-training a language model with a latent retrieval step so that retrieval
is learned from the language modelling objective rather than from supervised
question-answer pairs.

Subsequent work largely decoupled the components again. Izacard and Grave (2021)
showed with Fusion-in-Decoder that encoding many retrieved passages independently
and fusing them in the decoder scales better than concatenating them into a single
encoder input, and Izacard et al. (2023) extended this to few-shot settings with
Atlas. The dominant contemporary deployment pattern is looser still: a frozen
embedding model, a frozen generator, and a retrieval pipeline assembled from
off-the-shelf components. Gao et al. (2023) survey this landscape and distinguish
naive, advanced and modular RAG, a taxonomy whose main value is in making explicit
how much of a modern system sits in the retrieval pipeline rather than in either
model.

## 2.2 Sparse, dense and hybrid retrieval

Sparse lexical retrieval remains a strong baseline and is not a legacy method.
BM25, whose derivation and parameterisation Robertson and Zaragoza (2009) set out
in full, scores a document by a saturating function of query term frequency,
discounted by document length and weighted by inverse document frequency. Its
strength is exact term matching, which matters disproportionately for rare and
technical vocabulary: an unusual protein name, a statute number, or a product code
is retrieved reliably by BM25 and unreliably by a dense encoder whose training
distribution did not contain it.

Dense retrieval, by contrast, matches on distributional similarity and handles
paraphrase, synonymy and vocabulary mismatch, which BM25 cannot. Reimers and
Gurevych (2019) established the practical sentence-embedding recipe with
Sentence-BERT; Xiong et al. (2021) showed with ANCE that the choice of negatives
during contrastive training dominates the resulting quality, and that negatives
sampled from an approximate nearest neighbour index over the whole corpus are far
more informative than in-batch negatives. Chen et al. (2024) report multilingual,
multi-granularity embeddings in BGE-M3 that combine dense, sparse and multi-vector
retrieval in a single model.

Because the two families fail in complementary ways, hybrid retrieval is standard
practice. The combination can be performed on scores or on ranks. Score-level
fusion requires normalisation, since BM25 scores are unbounded and cosine
similarities lie in a fixed interval; per-query min-max normalisation followed by a
weighted sum is the common approach and is the one used in the accompanying project.
Rank-level fusion avoids the normalisation problem entirely, and Cormack et al.
(2009) showed that Reciprocal Rank Fusion, which sums the reciprocals of a
document's ranks across systems, outperforms both Condorcet fusion and learned
combination methods while requiring no tuning and no score calibration. That
RRF remains competitive with learned alternatives fifteen years later is a
useful caution about the marginal value of sophistication at this stage of the
pipeline.

A third family sits between sparse and dense. Khattab and Zaharia (2020) proposed
ColBERT, which represents a query and a document as sets of token-level embeddings
and scores them by a late-interaction MaxSim operator, retaining term-level
granularity while remaining precomputable and indexable; Santhanam et al. (2022)
reduced its storage cost substantially in ColBERTv2. Late interaction is relevant
to this review for a structural reason developed in Section 4.4: it decomposes a
similarity score into a sum of local contributions rather than computing a single
global inner product, and that decomposition is precisely what allows a scoring
function to reorder results that a single global inner product cannot.

## 2.3 Reranking

Retrieval is almost always followed by reranking, on the argument that an expensive
scoring function can be applied to a shortlist of a hundred candidates even if it
cannot be applied to a corpus of millions. Nogueira and Cho (2019) showed that a
cross-encoder, in which the query and passage are concatenated and jointly encoded
so that full attention operates across both, substantially improves precision at the
top of the ranking; Nogueira et al. (2020) reframed the same task as sequence
generation with monoT5. Cross-encoders are the strongest reranking family and are
also the most expensive, since no part of the computation can be precomputed.

Reranking for relevance alone is not sufficient when the output is a fixed-size
context window, because the top-ranked passages of a good retriever are frequently
near-duplicates of one another. Carbonell and Goldstein (1998) introduced Maximal
Marginal Relevance, which greedily selects the candidate maximising a weighted
difference between its relevance to the query and its maximum similarity to the
already-selected set. MMR is a greedy heuristic for what is in fact a combinatorial
selection problem, and its formulation as such is the direct ancestor of the QUBO
selection objective discussed in Section 6.4.

## 2.4 Evaluation, and the strong-baseline problem

Thakur et al. (2021) introduced BEIR as a heterogeneous zero-shot benchmark, and its
central finding reframed the field: dense retrievers that outperformed BM25 on the
datasets they were trained on frequently underperformed it when transferred, and
BM25 remained the strongest single method on average across the suite. The
practical consequence for any new ranking proposal is that a comparison against
dense retrieval alone is uninformative, and a comparison against an untuned lexical
baseline is misleading.

SciFact, from Wadden et al. (2020), is the corpus used in the accompanying project:
roughly five thousand scientific abstracts with claim-style queries and expert
relevance judgments. It is a demanding setting for a new ranking method, because
claims and their supporting abstracts share technical vocabulary heavily and lexical
retrieval consequently performs very well. A tuned hybrid baseline reaches recall@10
of roughly 0.84 and nDCG@10 of roughly 0.72 on it, which leaves little headroom.
Choosing such a corpus
makes an improvement harder to demonstrate and correspondingly harder to
manufacture, and we regard that as the right trade-off for a project whose central
risk is a component that does nothing.

The metrics themselves are standard. Järvelin and Kekäläinen (2002) defined
normalised discounted cumulative gain, which discounts gain logarithmically by rank
and normalises against the ideal ordering. Because retrieval metrics are averaged
over a small number of queries, differences between systems are frequently within
sampling noise, and reporting a mean without an interval is not adequate. Sakai
(2006) examined bootstrap methods for retrieval evaluation specifically; the paired
bootstrap, resampling queries with replacement and recomputing the per-query
difference, is the appropriate procedure when two systems are evaluated on the same
query set, and is what the accompanying project uses.

## 2.5 Known failure modes

Chen et al. (2024) benchmark large language models under retrieval augmentation and
identify four distinct capabilities that a RAG system requires and that models
possess unevenly: noise robustness, negative rejection, information integration,
and counterfactual robustness. Their finding on negative rejection is the most
consequential for this review. When the retrieved context contains no answer,
models frequently answer anyway rather than abstaining, which means that a retrieval
failure is converted into a fluent and confident false statement rather than into a
visible error.

This has a direct bearing on Section 7. If a system cannot reliably decline to
answer from irrelevant context, then an attacker who can insert passages into the
retrieved set has a reliable channel into the generated output, and the security of
the retrieval stage becomes a property of the system as a whole rather than a
peripheral concern. Asai et al. (2024) address the same weakness from the model
side with Self-RAG, training a model to retrieve adaptively and to emit critique
tokens assessing whether its output is supported by the retrieved evidence.

[[PAGEBREAK]]

# 3. Quantum Computing Foundations Relevant to Retrieval

## 3.1 States, amplitudes and measurement

The standard reference is Nielsen and Chuang (2010), and only the elements that bear
on retrieval are summarised here. The state of an $n$-qubit register is a unit
vector in a complex Hilbert space of dimension $2^n$, written as a superposition
over computational basis states with complex amplitudes. Two features distinguish
this from a classical probability vector. The amplitudes are complex, so each
carries a phase as well as a magnitude, and phases can cancel; this is interference,
and it is the mechanism behind every quantum algorithm that achieves a genuine
advantage. Second, the state is not directly observable. Measurement in the
computational basis returns one basis state with probability equal to the squared
modulus of its amplitude, and destroys the superposition.

The second point is a hard constraint on retrieval applications. A quantum register
of thirty qubits holds over a billion amplitudes, but a single measurement extracts
one index. Any algorithm that requires reading out many amplitudes must repeat the
whole computation, and the sampling cost cancels the apparent advantage of the large
state space. Useful quantum algorithms are therefore those whose answer is a single
value, or those that concentrate amplitude onto the states of interest before
measurement, which is exactly what Grover's algorithm does.

## 3.2 Encoding classical data

A retrieval algorithm must first get the data in. Schuld and Petruccione (2018)
catalogue the standard encodings, and the choice among them determines whether a
proposed advantage survives.

Basis encoding maps a bit string to the corresponding computational basis state.
It is simple and wasteful, using one qubit per bit.

Angle encoding maps each feature to a rotation angle on its own qubit, requiring
$d$ qubits for $d$ features and a circuit of depth independent of $d$. Schuld et
al. (2021) analyse how such an encoding determines the class of functions a
variational model can represent, showing that the model computes a truncated Fourier
series in the input whose accessible frequencies are fixed by the encoding, so that
the encoding, and not the trainable part of the circuit, bounds expressivity.

Amplitude encoding maps a normalised $d$-dimensional real vector onto the amplitudes
of $\lceil \log_2 d \rceil$ qubits. It is the encoding relevant to this review,
because it makes the inner product between two encoded vectors directly accessible:
the overlap of two amplitude-encoded states equals the inner product of the vectors,
and the squared modulus of that overlap, which is what a measurement yields, equals
the squared cosine similarity. The logarithmic qubit count is what makes the
encoding attractive and also what makes it deceptive, because preparing an arbitrary
amplitude-encoded state requires a circuit whose depth is linear in $d$ in general.
The exponential compression is in qubits, not in operations.

This is the state preparation bottleneck, and it is where most claimed advantages in
quantum machine learning are lost. Aaronson (2015) sets out the general form of the
problem for the HHL linear systems algorithm of Harrow et al. (2009): the
exponential speed-up holds only under assumptions about efficient state
preparation and about the form of the desired output, and applications that ignore
those assumptions are not entitled to the speed-up. Any retrieval proposal that
amplitude-encodes document vectors inherits this caveat in full, and a proposal that
also simulates the circuit classically, as the accompanying project does, pays the
$O(2^n)$ simulation cost on top.

## 3.3 NISQ constraints and barren plateaus

Preskill (2018) named the current era noisy intermediate-scale quantum, describing
devices with tens to hundreds of noisy qubits, no error correction, and coherence
budgets that limit circuit depth. Bharti et al. (2022) review the algorithms
designed for this regime; Cerezo et al. (2021) review variational quantum algorithms
specifically, in which a shallow parameterised circuit is optimised by a classical
outer loop, so that circuit depth is traded for a larger number of circuit
evaluations. Peruzzo et al. (2014) demonstrated the pattern on photonic hardware
and Kandala et al. (2017) on superconducting qubits.

The principal obstacle to scaling variational methods was identified by McClean et
al. (2018): for randomly initialised parameterised circuits, gradients of the cost
function vanish exponentially in the number of qubits, a phenomenon they named the
barren plateau. Its practical effect is that a variational circuit large enough to
be interesting may be untrainable, because the optimiser has no gradient signal to
follow. Two consequences bear on this review. First, variational quantum models
must be constructed with structure that avoids the plateau rather than initialised
randomly at scale, which limits the model classes available. Second, and more
narrowly, a phase-parameterised similarity function of the kind discussed in Section
4 is trainable when its gradient is derived analytically and computed on the
classical simulator, because the plateau is a property of estimating gradients from
measurement outcomes on a device.

## 3.4 What a classical simulation can and cannot demonstrate

A statevector simulator maintains the full $2^n$-dimensional amplitude vector in
classical memory. It therefore reproduces the algorithm's output exactly, up to
floating-point error, and reproduces nothing about its cost. This distinction is
not a technicality, and it is the source of the second major error pattern in the
applied literature: a paper reports that a quantum-enhanced retrieval system is
faster, when what was measured was a classical simulation of a quantum circuit
running alongside a classical baseline, an arrangement in which the simulated system
cannot be faster and the reported timing must be measuring something else.

What a simulator can legitimately establish is threefold. It can verify that an
algorithm is correct as specified. It can measure quantities defined in the
algorithm's own cost model rather than in seconds, of which the number of oracle
queries in an amplitude amplification routine is the canonical example, since that
count is a property of the algorithm and is identical on a simulator and on
hardware. And it can measure the quality of the output, such as how close a
variational optimiser gets to the true optimum of the objective it was given, which
is a question about the algorithm and not about the machine.

[[PAGEBREAK]]

# 4. Quantum and Quantum-Inspired Information Retrieval

## 4.1 The geometry of information retrieval

Van Rijsbergen (2004) proposed that the mathematics of quantum theory, rather than
its physics, provides a unifying formalism for information retrieval. The
observation is that the vector space model, probabilistic relevance models, and
logical models of retrieval can each be expressed in the language of Hilbert spaces,
projectors and density operators, and that doing so exposes relationships between
them that their native formulations obscure. Melucci (2015) develops this position
at length, and Piwowarski et al. (2010) give a probabilistic framework for
information retrieval built on quantum probability, representing information needs
as density operators and documents as subspaces.

It is worth being precise about what this programme claims, because it is frequently
overstated by later citers. It is a claim about representational adequacy, not about
computational speed. No quantum hardware is involved and none is required. The
argument is that quantum probability, which is defined over subspaces of a Hilbert
space rather than over subsets of a sample space, can express phenomena such as
interference between query interpretations and order effects in relevance judgments
that classical probability handles awkwardly. Uprety et al. (2020) survey the
resulting body of work in ACM Computing Surveys and are appropriately measured about
its empirical payoff.

## 4.2 Quantum language models

Sordoni et al. (2013) gave the most influential concrete instantiation, the Quantum
Language Model, which represents a query or document as a density matrix over term
projectors rather than as a multinomial distribution over terms. The off-diagonal
entries of the density matrix encode term dependency, so compound concepts are
represented without explicitly enumerating n-grams, and the authors report
improvements over unigram language models on standard collections.

Zhang et al. (2018) built an end-to-end neural quantum-like language model and
applied it to question answering. Li et al. (2019) proposed the Complex-valued
Network for Matching, representing words as complex-valued vectors in which
amplitude carries lexical weight and phase carries a distinct semantic component,
composing them into sentence-level density matrices and measuring against trainable
projectors. Wang et al. (2019) develop a related semantic Hilbert space
representation. The stated advantage of these models is interpretability: components
of the architecture correspond to identifiable quantum-mechanical objects, and the
authors argue this makes the resulting matching function more transparent than an
opaque neural scorer.

## 4.3 The role of phase

The recurring innovation across this family is the introduction of phase. A
classical embedding assigns each dimension a real magnitude; a complex-valued
embedding assigns each dimension a magnitude and an angle. The intuition offered is
that magnitude encodes how strongly a term is present and phase encodes something
about its sense or role, so that two documents containing the same terms with the
same weights but in different senses can be distinguished, which a real-valued bag
of weights cannot do.

The intuition is sound in principle, and phase does add representational capacity.
Whether it adds capacity *that changes a ranking* is a separate question, and it is
the question the literature answers least clearly.

## 4.4 The rank-equivalence problem

Consider the most natural quantum similarity score. Amplitude-encode a normalised
query vector $q$ and a normalised document vector $d$, introduce a per-dimension
phase $\theta_i$, and take the squared modulus of the overlap:

> $K(q, d) = \left| \sum_i q_i d_i e^{i\theta_i} \right|^2$

This is the fidelity between the two encoded states, it is what a swap test or an
inversion test measures, and it is the form used, with variations, throughout the
quantum kernel literature. At $\theta = 0$ it reduces to $\left( \sum_i q_i d_i
\right)^2$, which is the squared cosine similarity.

Squaring is a monotone increasing function on non-negative arguments. A ranking is
invariant under monotone transformations of the score. Therefore, for non-negative
embeddings and untrained phases, this quantum kernel produces *exactly the ranking
that cosine similarity produces*. It is not an approximation of the classical
ranking, and it is not a slightly different ranking; it is the same permutation.

The consequence is uncomfortable and, in our reading of this literature,
insufficiently confronted. A system that amplitude-encodes its vectors, computes
fidelity, and reports improved retrieval over a cosine baseline has either trained
the phases to something non-zero, or has obtained its improvement from another
component, or has a baseline that was not actually cosine. The first is a genuine
contribution; the second and third are attribution errors. Papers in this area
often do not report the diagnostic that would distinguish them, which is a rank
correlation between the quantum ranking and the classical ranking it is claimed to
improve on. A Kendall $\tau$ of 0.999 against cosine would settle the matter
immediately, and it is almost never reported.

Two structural escapes from the problem exist, and they are worth stating precisely
because they define what a non-vacuous quantum ranking function must do.

The first is trained non-zero phases. If the $\theta_i$ are fitted to a retrieval
objective, the score is no longer a monotone function of the cosine and can reorder.
This is a real escape, but it comes with a caveat that the literature tends not to
state: a model with $d$ free phase parameters fitted on a training set much smaller
than $d$ will overfit, and any improvement must be demonstrated on held-out data
with the parameter count reported alongside.

The second escape is structural and does not require training at all. Partition the
$d$ dimensions into $G$ disjoint blocks and take a weighted sum of per-block
fidelities:

> $K(q, d) = \sum_g w_g \left| \sum_{i \in g} q_i d_i e^{i\theta_i} \right|^2$

At $\theta = 0$ this equals $\sum_g w_g S_g^2$ where $S_g$ is the partial inner
product over block $g$. With uniform weights this is $\sum_g S_g^2$, and by the
Cauchy-Schwarz inequality $\left( \sum_g S_g \right)^2 \le G \sum_g S_g^2$ with
equality only when all $S_g$ are equal. The sum of squares is therefore *not* a
monotone function of the square of the sum. Two documents with the same total
cosine similarity but different distributions of that similarity across blocks
receive different scores, with concentrated agreement scoring above diffuse
agreement. The ranking genuinely differs from cosine, without any parameter being
fitted.

This is the same structural move that ColBERT makes for a different reason. Late
interaction scores by summing per-token maxima rather than by a single global inner
product, and it is the decomposition, not the quantum interpretation, that provides
the additional discriminative power. Recognising the two as instances of one
principle clarifies what the quantum framing contributes: it supplies a principled
reason to introduce phases into the local terms, which the classical late-interaction
formulation has no natural way to express.

## 4.5 Critical assessment

Three observations conclude this section.

The quantum-inspired retrieval literature has produced genuine representational
contributions, in the density-matrix treatment of term dependence and in the
complex-valued matching architectures, and these have been evaluated on standard
collections with reported gains. The gains are typically modest and the baselines
are typically not the strongest available at the time of publication.

The rank-equivalence problem is a specific, checkable defect that a reader can test
for and that authors can pre-empt with a single reported statistic. Its persistence
suggests that the diagnostic is not part of the field's standard reporting practice,
and we would argue that it should be: any paper proposing a fidelity-based ranking
function should report the rank correlation of its score against the classical
similarity it reduces to at zero phase.

Finally, none of this work requires quantum hardware, and most of it does not claim
to. It is quantum-inspired in the accurate sense of borrowing a mathematical
formalism. Conflating this with quantum computation is a category error that the
primary sources generally avoid and that secondary citations frequently commit.

[[PAGEBREAK]]

# 5. Quantum Kernel Methods

## 5.1 Feature Hilbert spaces

Schuld and Killoran (2019) and Havlicek et al. (2019), published concurrently in
Physical Review Letters and Nature respectively, established the framing that
dominates quantum machine learning for classical data. A quantum feature map takes
a classical input $x$ to a quantum state $|\phi(x)\rangle$, and the inner product
between two such states defines a kernel $k(x, x') = \left| \langle \phi(x) |
\phi(x') \rangle \right|^2$. Because kernel methods depend on the data only through
such inner products, a quantum device that can estimate this overlap can be used as
the kernel of a classical support vector machine, with the classical learning theory
carrying over unchanged.

Schuld (2021) sharpened this into a structural statement: supervised quantum machine
learning models on classical data *are* kernel methods, and a variational quantum
classifier is a linear model in the feature space induced by its encoding.
This is clarifying in both directions. It brings the guarantees of kernel theory
into quantum machine learning, and it bounds expectations, since the trainable
circuit cannot enlarge the feature space that the encoding fixed.

## 5.2 Fidelity kernels and their classical shadows

The kernel most commonly realised in practice is the fidelity kernel above, and it
is exactly the object discussed in Section 4.4. This connection is the reason
Section 4.4's argument can be stated as a theorem rather than an observation: for an
amplitude encoding with zero phases and non-negative inputs, the induced fidelity
kernel is the squared linear kernel, and a squared linear kernel induces the same
ranking as a linear one.

The literature on quantum kernels is therefore directly relevant to whether a
quantum retrieval score can help. The determining factor is the encoding. An
encoding whose induced kernel has an efficiently computable classical closed form
provides no computational advantage, whatever its quantum implementation; the
retrieval application inherits this immediately, since a kernel that equals
$\cos^2$ is not merely classically computable but is the classical similarity
already in use.

## 5.3 When quantum kernels can help

Huang et al. (2021) give the most careful treatment of when a quantum model can
outperform a classical one on classical data. Their analysis separates the question
into the expressive power of the quantum feature space and the amount of training
data available, and their central result is that access to data changes the picture
substantially: many problems for which a quantum advantage might be conjectured on
complexity-theoretic grounds admit efficient classical learning once training data
is available, because the learner need only match the target on the data
distribution rather than everywhere. They also give a projected quantum kernel
construction that measures local observables rather than global state overlap, and
show it generalises better than the global fidelity kernel.

The projected kernel construction is directly analogous to the block-local kernel of
Section 4.4. Both replace a single global overlap with a sum of local overlaps, and
both do so for the same underlying reason: a global fidelity kernel on
high-dimensional data becomes concentrated, in the sense that overlaps between
distinct inputs shrink towards a constant as dimension grows, which destroys the
kernel's ability to discriminate. Locality restores discrimination. That an
argument from generalisation in quantum machine learning and an argument from
Cauchy-Schwarz in retrieval ranking converge on the same architectural change is, we
think, the most useful theoretical connection this review draws.

Liu et al. (2021) provide the complementary positive result, a rigorous quantum
speed-up for a supervised learning problem based on discrete logarithms, which
establishes that quantum kernels can offer provable advantages. The problem is
constructed for the purpose and is not a natural learning task, which is the point:
it delimits what has actually been proven from what is hoped for.

## 5.4 Dequantisation

The most important corrective in this literature came from outside it. Tang (2019)
showed that the quantum recommendation systems algorithm of Kerenidis and Prakash,
which had been presented as an exponential speed-up over classical methods,
admits a classical algorithm with only polynomially worse scaling once the classical
algorithm is granted the same sampling access to its input that the quantum
algorithm was granted to its state preparation. The apparent exponential separation
was an artefact of comparing a quantum algorithm with strong input assumptions
against a classical algorithm with weak ones.

This result and the dequantisation literature it opened impose a discipline on any
claim of quantum advantage for a data-driven task. The comparison must be
like-for-like in input access, and a speed-up that depends on an unstated assumption
about how the data is made available is not a speed-up. For a retrieval system this
is the operative constraint: if document vectors must be prepared as quantum states
from a classical store, the preparation cost belongs in the accounting, and the
overwhelming majority of proposed retrieval speed-ups do not survive its inclusion.

[[PAGEBREAK]]

# 6. Search and Optimisation Subroutines

## 6.1 Grover's algorithm and the query model

Grover (1996) gave an algorithm that finds a marked item among $N$ unstructured
items using $O(\sqrt{N})$ evaluations of a black-box function that recognises the
marked item, against the $\Theta(N)$ evaluations a classical algorithm requires. The
mechanism is amplitude amplification: starting from a uniform superposition, repeated
application of an oracle that phase-flips marked states followed by a reflection
about the mean rotates the state vector towards the marked subspace, so that
measurement returns a marked item with high probability after approximately
$\frac{\pi}{4}\sqrt{N/M}$ iterations for $M$ marked items. Brassard et al. (2002)
generalised the construction to arbitrary initial state preparations and to
estimating $M$ rather than finding an element. Bennett et al. (1997) proved the
matching lower bound, establishing that the quadratic speed-up is optimal in the
query model and thereby closing off hope of an exponential improvement for
unstructured search.

The phrase *in the query model* carries the entire content of the result, and it is
the most frequently dropped qualification in applied citations of Grover. The
theorem counts oracle evaluations. It says nothing about the cost of implementing
the oracle, nothing about the cost of preparing the initial superposition, and
nothing about seconds. When the items to be searched are records in a classical
database, the oracle must be constructed from that data, and the construction cost
is generally linear in $N$, which eliminates the advantage. Grover's algorithm is
therefore properly applied where the oracle is a computation rather than a lookup,
so that marked items are recognised by evaluating a predicate rather than by
consulting a store.

## 6.2 Unknown numbers of solutions

The iteration count above depends on $M$, which is generally unknown in advance, and
over-rotation is a real failure mode: continuing past the optimal iteration count
rotates the state away from the marked subspace again, so more iterations can make
the outcome strictly worse. Boyer et al. (1998) resolved this with an exponentially
increasing schedule of randomly chosen iteration counts, which finds a solution in
expected $O(\sqrt{N/M})$ queries without knowing $M$, and also give the tight
analysis of success probability as a function of iteration count. For retrieval
applications where the number of documents above a relevance threshold is not known
before searching, this variant rather than textbook Grover is the correct reference.

## 6.3 Applying amplitude amplification to retrieval

The honest form of a Grover-based retrieval claim is narrow. Given a shortlist of
candidates already scored by a classical retriever, and a threshold predicate that
marks candidates whose score exceeds some value, amplitude amplification identifies a
marked candidate in $O(\sqrt{N/M})$ oracle queries where classical scanning requires
$O(N/M)$ in expectation. This is a correct statement about query complexity, it is
verifiable on a simulator because the query count is machine-independent, and it is
the only claim the construction supports.

What it does not support is a latency claim. On a classical simulator, each Grover
iteration manipulates a $2^n$-dimensional amplitude vector, so simulating a
$\sqrt{N}$-query search costs more arithmetic than performing the $N$ classical
comparisons it replaces. The correct presentation reports the oracle-query reduction
and the simulation overhead as two separate quantities, adjacent to one another, so
that no reader can mistake the first for a timing result. The accompanying project
adopts this convention deliberately, and its results tables carry both columns.

A further limitation deserves statement. The shortlist must already have been
scored classically for the threshold oracle to exist. Amplitude amplification is
therefore operating downstream of the retrieval work, on a candidate set small
enough that the classical scan was never the bottleneck. It is a demonstration of a
complexity property on a real workload, not a component that makes the pipeline
faster, and describing it as the latter would be the error this review has already
identified twice.

## 6.4 The Quantum Approximate Optimization Algorithm

Farhi et al. (2014) introduced QAOA for approximate combinatorial optimisation. A
problem is encoded as a cost Hamiltonian diagonal in the computational basis, so that
each basis state's eigenvalue is the objective value of the corresponding assignment.
A trial state is prepared by alternating $p$ layers of cost-Hamiltonian phase
evolution and transverse-field mixing, applied to a uniform superposition, with the
$2p$ evolution times as free parameters. A classical optimiser adjusts the
parameters to minimise the measured expectation of the cost, and the final state is
sampled to obtain candidate solutions. As $p \to \infty$ the ansatz can express the
adiabatic path and recover the optimum; at small $p$ it is a heuristic.

The algorithm's practical behaviour has been examined carefully and the picture is
mixed. Zhou et al. (2020) analyse performance and parameter structure at low depth
and give heuristic strategies for parameter setting that substantially reduce the
optimisation burden. Against this, Hastings (2019) showed that classical local
algorithms match or exceed QAOA at $p = 1$ on the problem families where it had been
studied, and Bravyi et al. (2020) identified symmetry-protection obstacles that
prevent QAOA from outperforming simple classical algorithms on certain instances
regardless of depth. Farhi and Harrow (2016) argue for the algorithm's
hardness-of-simulation properties, which is a different and weaker claim than
optimisation advantage. The reasonable summary is that QAOA is a well-motivated
heuristic whose advantage over good classical heuristics is unproven and, at low
depth, has been shown absent in specific settings.

Two practical points bear on any implementation. First, the classical optimiser
matters. Powell's COBYLA (1994), a derivative-free trust-region method using linear
interpolation models, is the common choice for QAOA because the objective is
evaluated by sampling and gradients are expensive and noisy; multiple restarts are
necessary because the parameter landscape is non-convex. Second, constraint
handling. Constraints are typically imposed as quadratic penalty terms, which means
the search space includes infeasible assignments and a measured sample may violate
the constraint. Reporting the fraction of samples that are feasible is therefore
part of characterising the method, not an optional diagnostic.

## 6.5 Context selection as a QUBO

Lucas (2014) catalogues Ising formulations for a wide range of NP-hard problems, and
Glover et al. (2019) give a practical tutorial on QUBO modelling. The relevant
instance for retrieval is subset selection with a diversity penalty. Given
candidate relevance scores $r_i$ and pairwise similarities $s_{ij}$, and a context
budget of $k$ passages, the objective

> $\min_x \; -\sum_i r_i x_i \;+\; \lambda \sum_{i<j} s_{ij} x_i x_j \;+\; \mu \left( \sum_i x_i - k \right)^2$

trades total relevance against total redundancy subject to a soft cardinality
constraint. This is a quadratic unconstrained binary optimisation problem in the
form QAOA consumes directly, and it is a direct combinatorial statement of what
Maximal Marginal Relevance (Carbonell and Goldstein, 1998) approximates greedily.
The formulation is where a genuine, if modest, argument for a quantum subroutine in
retrieval can be made: the selection problem really is combinatorial, the greedy
classical method really is a heuristic with no optimality guarantee, and at the
scale of a context window the problem is small enough to be tractable on near-term
devices.

An important methodological point follows from that small scale. For $n \le 20$
candidates the exact optimum can be found by enumeration in negligible time, so
QAOA's solution quality can be measured against the true optimum rather than against
another heuristic. Any implementation at this scale that reports approximation
quality without computing the exact optimum has chosen not to make the strongest
available comparison.

The quality metric itself requires care, and this review notes the issue because it
is easy to get wrong. The natural definition, achieved objective divided by optimal
objective, is not affine-invariant, and the objective above changes sign with the
balance between relevance and similarity: it is negative when relevance dominates
and positive when candidates are weakly relevant and mutually similar. In the
positive regime the ratio rewards worse solutions and can exceed one, producing the
absurd appearance of a heuristic beating an exact optimum over the same feasible
set. The normalised form $q = (f_{\text{worst}} - f) / (f_{\text{worst}} -
f_{\text{opt}})$, computed over the same feasible set, is invariant under affine
reparameterisation of the objective and is bounded in $[0, 1]$ by construction.

## 6.6 A reporting discipline for simulated quantum subroutines

Drawing Sections 6.1 to 6.5 together, four reporting rules follow, and this review
states them explicitly because their violation accounts for a large share of the
weak claims in applied quantum retrieval:

1. Report complexity in the units the theorem is stated in. Grover gives oracle
   queries. Present them as oracle queries.
2. Report simulation cost separately and adjacently, so that a complexity result
   cannot be read as a latency result.
3. Compare against the strongest available classical alternative, which at
   context-window scale means the exact optimum and not a greedy heuristic.
4. Report the diagnostic that would falsify the claim: rank correlation against the
   classical score for a ranking function, and feasible-sample fraction for a
   penalty-constrained optimiser.

[[PAGEBREAK]]

# 7. Security of Retrieval-Augmented Systems

## 7.1 Corpus poisoning

A retrieval-augmented system trusts its corpus, and in many deployments the corpus
is not fully trusted: it may contain scraped web content, user uploads, or documents
from a shared store with many writers. An adversary who can insert text into the
corpus can attempt to have that text retrieved and thereby placed in the model's
context.

Zhong et al. (2023) established the strongest form of the attack on the retrieval
stage. Using gradient-based token substitution in the style of HotFlip (Ebrahimi et
al., 2018), they optimise adversarial passages to sit near the centroid of a cluster
of query embeddings, so that a small number of inserted passages are retrieved for a
broad range of unseen queries. The attack transfers across queries and, to a
degree, across retrievers. Its requirement is gradient access to the embedding
model, which is available for open-weight encoders and not for API-only ones. Any
black-box attack, restricted to forward passes and heuristic search, is strictly
weaker, and a defence evaluated only against a black-box attack has been tested
against the easier case.

Zou et al. (2025) extend the threat model to the generation stage in PoisonedRAG,
constructing injected passages that must satisfy two conditions simultaneously:
retrievability for the target query, and sufficiency to induce a specific attacker
chosen answer once retrieved. They show that a handful of injected passages per
target question is enough to control the answer at high rates, which reframes corpus
poisoning from a retrieval-quality nuisance into an output-integrity attack.

## 7.2 Indirect prompt injection

Prompt injection proper was characterised by Perez and Ribeiro (2022), who showed
that instructions embedded in user input can override a system prompt. The indirect
variant is the one that matters for RAG. Greshake et al. (2023) showed that
instructions embedded in *retrieved* content are executed by LLM-integrated
applications, so that an attacker who controls a document the system will retrieve
controls, to a substantial degree, what the system does. Liu et al. (2023) provide
a systematic evaluation across deployed applications. The OWASP Top 10 for Large
Language Model Applications lists prompt injection as its first entry, and treats
indirect injection through retrieved content as the more serious form on the
reasoning that the attacker need not be a user of the system at all.

The structural problem is that a language model receives instructions and data in
the same channel, with no mechanism at the architectural level for distinguishing
them. This is the same class of defect as SQL injection, and the analogy is
instructive in what it suggests about remedies: SQL injection was solved by
parameterised queries, which separate code from data at the protocol level, and no
equivalent separation currently exists for language model prompts.

Boucher et al. (2022) demonstrate a complementary and often overlooked surface.
Imperceptible perturbations using invisible Unicode characters, homoglyphs, and
bidirectional control characters produce text that a human reader and an automated
filter see differently, so a passage that appears innocuous when displayed can carry
an instruction that the model receives. The Unicode Tags block is a particularly
direct instance, since characters in it are typically invisible in rendered text yet
tokenise to content the model can act on. Any input sanitisation for a
retrieval-augmented system that operates on rendered appearance rather than on code
points is therefore incomplete.

## 7.3 Defences and their limits

Hines et al. (2024) propose spotlighting: marking retrieved content
unambiguously so the model can distinguish it from instructions, by delimiting it,
encoding it, or interleaving markers. They report substantial reductions in
injection success. The defence is a mitigation rather than a solution, since it
depends on the model's willingness to respect the marking, which is a behavioural
property and not an enforced one.

Pattern-based detection, scanning retrieved passages for imperative phrasing and
role markers before they enter the context, is cheap and catches unsophisticated
attacks. Its limitation is severe and predictable: an adversarial passage that
contains no instructions at all, and instead simply asserts a plausible-sounding
falsehood in the register of the surrounding corpus, presents no pattern to detect.
Against retrieval-stage poisoning of the Zhong et al. variety, and against the
fluent-mimicry passages of PoisonedRAG, a pattern detector is close to useless. It
is a defence-in-depth layer and must be reported as one.

Defences on the retrieval side are less explored. Diversity-aware selection has a
plausible mechanism against clustered injections, since attacks that insert several
mutually similar passages to raise the probability that at least one is retrieved
produce exactly the redundancy that a diversity penalty suppresses. To our
knowledge this has not been evaluated as a security control in the published
literature, and it is the specific hypothesis the accompanying project tests. We
note that the hypothesis is falsifiable and may simply be false, since an attack that
inserts a single well-optimised passage per target query produces no cluster to
suppress.

Es et al. (2024) provide RAGAS for reference-free evaluation of retrieval-augmented
pipelines, measuring faithfulness of the answer to the retrieved context and
relevance of the context to the question. Faithfulness metrics are relevant here in
a way that deserves emphasis: a successfully poisoned system may score *well* on
faithfulness, because the answer is faithful to the retrieved context and the
retrieved context is what the attacker supplied. Faithfulness measures grounding,
not truth, and using it as a safety metric confuses the two.

## 7.4 Evaluating defences honestly

Three evaluation practices follow from the above and are adopted in the accompanying
project.

The strength of the attack must be characterised, not merely its existence. A
defence result against a black-box embedding-optimisation attack is not a defence
result against Zhong et al. (2023), and the difference must be stated wherever the
numbers appear.

Detection rates must be reported per attack family and not only in aggregate. A
detector that flags one hundred per cent of instruction-injection passages and zero
per cent of topical-mimicry passages has an aggregate rate that depends entirely on
the mix of families in the test set, and the aggregate conceals the failure that
matters.

Finally, an ablation is required to attribute a defensive effect to a mechanism. If
a full pipeline admits fewer adversarial passages into its context than a baseline,
that difference could come from any of its components. Only a run with the
hypothesised mechanism disabled and everything else held fixed identifies the cause.

[[PAGEBREAK]]

# 8. Synthesis

## 8.1 What is established

Several things can be regarded as settled. Hybrid sparse-dense retrieval outperforms
either family alone, and lexical retrieval remains a strong baseline that must be
tuned rather than assumed weak (Robertson and Zaragoza, 2009; Thakur et al., 2021).
Quantum feature maps induce kernels, and variational quantum models on classical
data are kernel methods with the encoding fixing the feature space (Schuld and
Killoran, 2019; Schuld, 2021). Grover's quadratic speed-up in the query model is
optimal (Bennett et al., 1997). Retrieval-augmented systems are vulnerable to corpus
poisoning and to indirect prompt injection, at rates that make both practical
concerns rather than theoretical ones (Zhong et al., 2023; Greshake et al., 2023;
Zou et al., 2025).

## 8.2 What is contested

Whether quantum kernels offer an advantage on natural, as opposed to constructed,
learning problems remains open. Liu et al. (2021) establish that provable advantages
exist; Huang et al. (2021) establish that access to data narrows the space where
they can be expected; Tang (2019) and the dequantisation literature establish that
several prominent claimed advantages were artefacts of asymmetric input assumptions.
Whether QAOA outperforms good classical heuristics at accessible depths is likewise
unsettled, with Hastings (2019) and Bravyi et al. (2020) providing specific negative
results against Zhou et al. (2020)'s positive characterisation.

## 8.3 What is missing

Three gaps are visible in the intersection of these literatures.

First, the rank-equivalence diagnostic is absent from the quantum retrieval
literature. Papers proposing fidelity-based ranking functions do not, as a rule,
report the rank correlation between their score and the classical similarity their
score reduces to at zero phase, which is the single statistic that distinguishes a
contribution from a no-op. Section 4.4 argues that this should be standard practice
and that the reduction is often exact rather than approximate.

Second, complexity and simulation cost are routinely conflated. A convention of
reporting oracle queries and simulator wall-clock as separate, adjacent quantities
would remove the ambiguity at no cost, and its absence allows papers to imply timing
results that their experiments do not contain.

Third, retrieval-side defences against corpus poisoning are largely unevaluated.
Diversity-aware context selection has a clear mechanism against clustered
injections and, as far as we can determine, no published measurement, either
positive or negative.

## 8.4 Positioning of the accompanying project

The project this review supports is positioned against the three gaps above rather
than against a performance target, and its design commitments follow from them
directly.

It defines two gates before training the kernel. Gate A requires that the kernel's
ranking differ from cosine's by a Kendall $\tau$ below 0.995, which is the
rank-equivalence diagnostic of Section 4.4 made into a pass-fail condition. Gate B
requires that the resulting fusion improve mean reciprocal rank on held-out pairs,
which separates *reordering* from *improving*. A component that fails Gate A is a
no-op and a component that passes Gate A but fails Gate B is active and harmful, and
both outcomes are reportable results rather than failures of the experiment.

It adopts the block-local kernel of Section 4.4 rather than the global fidelity
kernel, on the structural argument from Cauchy-Schwarz, and this argument is
converted into a unit test: if the block kernel is ever found rank-equivalent to
cosine, the comparison is vacuous and the test fails loudly.

It reports Grover as oracle queries with simulation overhead in an adjacent column,
computes QAOA's exact optimum by enumeration at context-window scale, and uses the
affine-invariant quality metric of Section 6.5 rather than a ratio of objectives.

For the security arm it injects four attack families, states explicitly that its
embedding-optimisation family is black-box and therefore weaker than Zhong et al.
(2023), reports detection per family rather than in aggregate, and includes an
ablation with QAOA disabled so that any reduction in adversarial context occupancy
can be attributed to the redundancy penalty rather than to the pipeline as a whole.

The project makes no claim of a wall-clock speed-up, and its own measurements show
the full simulated pipeline to be substantially slower than its classical baseline.
That is stated as a property of the method rather than as a limitation to be
apologised for, since it follows from Section 3.4 and could not be otherwise.

# 9. Conclusion

The literature surveyed here supports a narrow but real space for quantum methods in
retrieval, and a much larger space of claims that will not survive scrutiny. The
narrow space has three properties. The scoring function must be structurally capable
of reordering results, which rules out global fidelity kernels on non-negative
embeddings with untrained phases and admits block-local or projected constructions.
Complexity claims must be made in the units the underlying theorems use, which
means oracle queries for amplitude amplification and solution quality against an
exact optimum for variational optimisation. And the comparison must be against a
tuned classical system with like-for-like input access, which is what the
dequantisation results require.

The security literature adds a fourth requirement that the quantum retrieval
literature has not yet absorbed: the retrieval stage is an attack surface, defences
must be attributed to mechanisms by ablation, and detection rates must be reported
per attack family because the aggregate conceals the failures that matter.

The most useful theoretical observation this review draws is the convergence noted in
Section 5.3. An argument from generalisation in quantum machine learning, which
motivates projected quantum kernels because global fidelity kernels concentrate in
high dimension, and an argument from Cauchy-Schwarz in retrieval ranking, which
shows that a sum of squared block overlaps is not monotone in the squared total
overlap, recommend the same architectural change for different reasons. Where two
independent lines of reasoning converge on one construction, that construction is
where the work should be done.

[[PAGEBREAK]]

# References

Aaronson, S. (2015). Read the fine print. *Nature Physics*, 11(4), 291-293.

Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2024). Self-RAG: Learning
to retrieve, generate, and critique through self-reflection. In *Proceedings of the
International Conference on Learning Representations (ICLR)*.

Bennett, C. H., Bernstein, E., Brassard, G., and Vazirani, U. (1997). Strengths and
weaknesses of quantum computing. *SIAM Journal on Computing*, 26(5), 1510-1523.

Bharti, K., Cervera-Lierta, A., Kyaw, T. H., Haug, T., Alperin-Lea, S., Anand, A.,
Degroote, M., Heimonen, H., Kottmann, J. S., Menke, T., Mok, W.-K., Sim, S., Kwek,
L.-C., and Aspuru-Guzik, A. (2022). Noisy intermediate-scale quantum algorithms.
*Reviews of Modern Physics*, 94(1), 015004.

Boucher, N., Shumailov, I., Anderson, R., and Papernot, N. (2022). Bad characters:
Imperceptible NLP attacks. In *Proceedings of the IEEE Symposium on Security and
Privacy (S&P)*, 1987-2004.

Boyer, M., Brassard, G., Hoyer, P., and Tapp, A. (1998). Tight bounds on quantum
searching. *Fortschritte der Physik*, 46(4-5), 493-505.

Brassard, G., Hoyer, P., Mosca, M., and Tapp, A. (2002). Quantum amplitude
amplification and estimation. *Contemporary Mathematics*, 305, 53-74.

Bravyi, S., Kliesch, A., Koenig, R., and Tang, E. (2020). Obstacles to variational
quantum optimization from symmetry protection. *Physical Review Letters*, 125(26),
260505.

Carbonell, J. and Goldstein, J. (1998). The use of MMR, diversity-based reranking
for reordering documents and producing summaries. In *Proceedings of the 21st Annual
International ACM SIGIR Conference on Research and Development in Information
Retrieval*, 335-336.

Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K.,
Roberts, A., Brown, T., Song, D., Erlingsson, U., Oprea, A., and Raffel, C. (2021).
Extracting training data from large language models. In *Proceedings of the 30th
USENIX Security Symposium*, 2633-2650.

Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K.,
McClean, J. R., Mitarai, K., Yuan, X., Cincio, L., and Coles, P. J. (2021).
Variational quantum algorithms. *Nature Reviews Physics*, 3(9), 625-644.

Chen, D., Fisch, A., Weston, J., and Bordes, A. (2017). Reading Wikipedia to answer
open-domain questions. In *Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (ACL)*, 1870-1879.

Chen, J., Lin, H., Han, X., and Sun, L. (2024). Benchmarking large language models
in retrieval-augmented generation. In *Proceedings of the AAAI Conference on
Artificial Intelligence*, 38(16), 17754-17762.

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. (2024). BGE
M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings
through self-knowledge distillation. In *Findings of the Association for
Computational Linguistics: ACL 2024*.

Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal rank fusion
outperforms Condorcet and individual rank learning methods. In *Proceedings of the
32nd International ACM SIGIR Conference on Research and Development in Information
Retrieval*, 758-759.

Ebrahimi, J., Rao, A., Lowd, D., and Dou, D. (2018). HotFlip: White-box adversarial
examples for text classification. In *Proceedings of the 56th Annual Meeting of the
Association for Computational Linguistics (ACL)*, 31-36.

Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman
and Hall, New York.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated
evaluation of retrieval augmented generation. In *Proceedings of the 18th Conference
of the European Chapter of the Association for Computational Linguistics (EACL):
System Demonstrations*, 150-158.

Farhi, E., Goldstone, J., and Gutmann, S. (2014). A quantum approximate optimization
algorithm. *arXiv preprint arXiv:1411.4028*.

Farhi, E. and Harrow, A. W. (2016). Quantum supremacy through the quantum
approximate optimization algorithm. *arXiv preprint arXiv:1602.07674*.

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., and Wang,
H. (2023). Retrieval-augmented generation for large language models: A survey.
*arXiv preprint arXiv:2312.10997*.

Glover, F., Kochenberger, G., and Du, Y. (2019). A tutorial on formulating and using
QUBO models. *4OR: A Quarterly Journal of Operations Research*, 17, 335-371.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., and Fritz, M. (2023).
Not what you've signed up for: Compromising real-world LLM-integrated applications
with indirect prompt injection. In *Proceedings of the 16th ACM Workshop on
Artificial Intelligence and Security (AISec)*, 79-90.

Grover, L. K. (1996). A fast quantum mechanical algorithm for database search. In
*Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC)*,
212-219.

Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. (2020). REALM:
Retrieval-augmented language model pre-training. In *Proceedings of the 37th
International Conference on Machine Learning (ICML)*, 3929-3938.

Harrow, A. W., Hassidim, A., and Lloyd, S. (2009). Quantum algorithm for linear
systems of equations. *Physical Review Letters*, 103(15), 150502.

Hastings, M. B. (2019). Classical and quantum bounded depth approximation
algorithms. *arXiv preprint arXiv:1905.07047*.

Havlicek, V., Corcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M.,
and Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature
spaces. *Nature*, 567(7747), 209-212.

Hines, K., Lopez, G., Hall, M., Zarfati, F., Zunger, Y., and Kiciman, E. (2024).
Defending against indirect prompt injection attacks with spotlighting. *arXiv
preprint arXiv:2403.14720*.

Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H., and
McClean, J. R. (2021). Power of data in quantum machine learning. *Nature
Communications*, 12(1), 2631.

Izacard, G. and Grave, E. (2021). Leveraging passage retrieval with generative
models for open domain question answering. In *Proceedings of the 16th Conference of
the European Chapter of the Association for Computational Linguistics (EACL)*,
874-880.

Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T.,
Dwivedi-Yu, J., Joulin, A., Riedel, S., and Grave, E. (2023). Atlas: Few-shot
learning with retrieval augmented language models. *Journal of Machine Learning
Research*, 24(251), 1-43.

Jarvelin, K. and Kekalainen, J. (2002). Cumulated gain-based evaluation of IR
techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.

Javadi-Abhari, A., Treinish, M., Krsulich, K., Wood, C. J., Lishman, J., Gacon, J.,
Martiel, S., Nation, P. D., Bishop, L. S., Cross, A. W., Johnson, B. R., and
Gambetta, J. M. (2024). Quantum computing with Qiskit. *arXiv preprint
arXiv:2405.08810*.

Johnson, J., Douze, M., and Jegou, H. (2021). Billion-scale similarity search with
GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M., and
Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small
molecules and quantum magnets. *Nature*, 549(7671), 242-246.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih,
W.-t. (2020). Dense passage retrieval for open-domain question answering. In
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP)*, 6769-6781.

Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*, 30(1-2),
81-93.

Kerenidis, I. and Prakash, A. (2017). Quantum recommendation systems. In
*Proceedings of the 8th Innovations in Theoretical Computer Science Conference
(ITCS)*, 49:1-49:21.

Khattab, O. and Zaharia, M. (2020). ColBERT: Efficient and effective passage search
via contextualized late interaction over BERT. In *Proceedings of the 43rd
International ACM SIGIR Conference on Research and Development in Information
Retrieval*, 39-48.

Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. In
*Proceedings of the International Conference on Learning Representations (ICLR)*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler,
H., Lewis, M., Yih, W.-t., Rocktaschel, T., Riedel, S., and Kiela, D. (2020).
Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in
Neural Information Processing Systems (NeurIPS)*, 33, 9459-9474.

Li, Q., Wang, B., and Melucci, M. (2019). CNM: An interpretable complex-valued
network for matching. In *Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics (NAACL-HLT)*, 4139-4148.

Liu, Y., Arunachalam, S., and Temme, K. (2021). A rigorous and robust quantum
speed-up in supervised machine learning. *Nature Physics*, 17(9), 1013-1017.

Liu, Y., Deng, G., Li, Y., Wang, K., Zhang, T., Liu, Y., Wang, H., Zheng, Y., and
Liu, Y. (2023). Prompt injection attack against LLM-integrated applications. *arXiv
preprint arXiv:2306.05499*.

Lucas, A. (2014). Ising formulations of many NP problems. *Frontiers in Physics*, 2,
5.

Malkov, Y. A. and Yashunin, D. A. (2020). Efficient and robust approximate nearest
neighbor search using hierarchical navigable small world graphs. *IEEE Transactions
on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., and Neven, H. (2018).
Barren plateaus in quantum neural network training landscapes. *Nature
Communications*, 9(1), 4812.

Melucci, M. (2015). *Introduction to Information Retrieval and Quantum Mechanics*.
Springer, Berlin.

Nielsen, M. A. and Chuang, I. L. (2010). *Quantum Computation and Quantum
Information: 10th Anniversary Edition*. Cambridge University Press, Cambridge.

Nogueira, R. and Cho, K. (2019). Passage re-ranking with BERT. *arXiv preprint
arXiv:1901.04085*.

Nogueira, R., Jiang, Z., Pradeep, R., and Lin, J. (2020). Document ranking with a
pretrained sequence-to-sequence model. In *Findings of the Association for
Computational Linguistics: EMNLP 2020*, 708-718.

OWASP (2025). *OWASP Top 10 for Large Language Model Applications*. Open Worldwide
Application Security Project.

Perez, F. and Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for
language models. In *NeurIPS ML Safety Workshop*.

Peruzzo, A., McClean, J., Shadbolt, P., Yung, M.-H., Zhou, X.-Q., Love, P. J.,
Aspuru-Guzik, A., and O'Brien, J. L. (2014). A variational eigenvalue solver on a
photonic quantum processor. *Nature Communications*, 5(1), 4213.

Piwowarski, B., Frommholz, I., Lalmas, M., and van Rijsbergen, K. (2010). What can
quantum theory bring to information retrieval? In *Proceedings of the 19th ACM
International Conference on Information and Knowledge Management (CIKM)*, 59-68.

Powell, M. J. D. (1994). A direct search optimization method that models the
objective and constraint functions by linear interpolation. In *Advances in
Optimization and Numerical Analysis*, 51-67. Springer, Dordrecht.

Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, 2, 79.

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods
in Natural Language Processing (EMNLP)*, 3982-3992.

Robertson, S. and Zaragoza, H. (2009). The probabilistic relevance framework: BM25
and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

Sakai, T. (2006). Evaluating evaluation metrics based on the bootstrap. In
*Proceedings of the 29th Annual International ACM SIGIR Conference on Research and
Development in Information Retrieval*, 525-532.

Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2022).
ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In
*Proceedings of the 2022 Conference of the North American Chapter of the Association
for Computational Linguistics (NAACL)*, 3715-3734.

Schuld, M. (2021). Supervised quantum machine learning models are kernel methods.
*arXiv preprint arXiv:2101.11020*.

Schuld, M. and Killoran, N. (2019). Quantum machine learning in feature Hilbert
spaces. *Physical Review Letters*, 122(4), 040504.

Schuld, M. and Petruccione, F. (2018). *Supervised Learning with Quantum Computers*.
Springer, Cham.

Schuld, M., Sweke, R., and Meyer, J. J. (2021). Effect of data encoding on the
expressive power of variational quantum-machine-learning models. *Physical Review
A*, 103(3), 032430.

Sordoni, A., Nie, J.-Y., and Bengio, Y. (2013). Modeling term dependencies with
quantum language models for IR. In *Proceedings of the 36th International ACM SIGIR
Conference on Research and Development in Information Retrieval*, 653-662.

Tang, E. (2019). A quantum-inspired classical algorithm for recommendation systems.
In *Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing
(STOC)*, 217-228.

Thakur, N., Reimers, N., Ruckle, A., Srivastava, A., and Gurevych, I. (2021). BEIR:
A heterogeneous benchmark for zero-shot evaluation of information retrieval models.
In *Advances in Neural Information Processing Systems (NeurIPS) Datasets and
Benchmarks Track*.

Uprety, S., Gkoumas, D., and Song, D. (2020). A survey of quantum theory inspired
approaches to information retrieval. *ACM Computing Surveys*, 53(5), 1-39.

van den Oord, A., Li, Y., and Vinyals, O. (2018). Representation learning with
contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

van Rijsbergen, C. J. (2004). *The Geometry of Information Retrieval*. Cambridge
University Press, Cambridge.

Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., and Hajishirzi,
H. (2020). Fact or fiction: Verifying scientific claims. In *Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 7534-7550.

Wallace, E., Feng, S., Kandpal, N., Gardner, M., and Singh, S. (2019). Universal
adversarial triggers for attacking and analyzing NLP. In *Proceedings of the 2019
Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2153-2162.

Wang, B., Li, Q., Melucci, M., and Song, D. (2019). Semantic Hilbert space for text
representation learning. In *Proceedings of the World Wide Web Conference (WWW)*,
3293-3299.

Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P., Ahmed, J., and
Overwijk, A. (2021). Approximate nearest neighbor negative contrastive learning for
dense text retrieval. In *Proceedings of the International Conference on Learning
Representations (ICLR)*.

Zhang, P., Niu, J., Su, Z., Wang, B., Ma, L., and Song, D. (2018). End-to-end
quantum-like language models with application to question answering. In
*Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1), 5666-5673.

Zhong, Z., Huang, Z., Wettig, A., and Chen, D. (2023). Poisoning retrieval corpora
by injecting adversarial passages. In *Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing (EMNLP)*, 13764-13775.

Zhou, L., Wang, S.-T., Choi, S., Pichler, H., and Lukin, M. D. (2020). Quantum
approximate optimization algorithm: Performance, mechanism, and implementation on
near-term devices. *Physical Review X*, 10(2), 021067.

Zou, W., Geng, R., Wang, B., and Jia, J. (2025). PoisonedRAG: Knowledge corruption
attacks to retrieval-augmented generation of large language models. In *Proceedings
of the 34th USENIX Security Symposium*.
