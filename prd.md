# Product Requirements Document  
**Automated Theorem-Proving Assistant with Claude-Powered Denoiser**

---

## 1. Executive Summary  
We’re building an end-to-end theorem proving assistant that:  
1. **Generates** candidate proofs using the SEDD model (from “Learning to Prove Theorems by Learning to Generate Theorems”).  
2. **Injects** controlled “noise” to simulate realistic proof errors.  
3. **Retrieves** relevant lemmas/theorems via RAG.  
4. **Denoises** and “re-logics” the proof with Claude.  
5. **Verifies** each step in a formal checker (e.g. Metamath, Lean).  

This hybrid pipeline leverages Claude’s language and reasoning capabilities to improve accuracy, interpretability, and robustness over purely neural provers.

---

## 2. Problem Statement  
- **Data scarcity**: Human‐authored proofs are limited, capping current provers’ performance.  
- **Hallucination & gaps**: Purely statistical proof generators can introduce logical gaps or invalid steps.  
- **Verification overhead**: Blindly trusting generated proofs leads to errors and manual debugging.

---

## 3. Proposed Solution  
Combine synthetic‐data proof generation (SEDD) with a Claude-powered “denoiser” in a noise-recovery loop to:  
- **Augment** proof corpora and boost coverage.  
- **Ground** each step in real lemmas via RAG.  
- **Correct** mistakes in a controlled, explainable way.  
- **Verify** with a formal system for end-to-end soundness.

---

## 4. Key Features  

| Feature                        | Description                                                                                   |
|--------------------------------|-----------------------------------------------------------------------------------------------|
| SEDD Model Integration         | Leverage the state-of-art SEDD generator for initial proof trees.                             |
| Noise Injection Module         | Apply tunable perturbations (drop, swap, reorder steps) to surface latent errors.            |
| Retrieval-Augmented Context    | Index a theorem/lemma corpus; retrieve top-k relevant items via Claude embeddings.           |
| Claude-Powered Denoiser        | Prompt Claude to correct and re-logicalize noisy proofs—step by step, citing lemmas.         |
| Formal Verification Loop       | Use Metamath/Lean to check each step; loop back on failures for retrial or escalation.       |
| User Interface                 | CLI and simple web UI for submitting theorems, visualizing proofs, and inspecting errors.    |
| Monitoring & Metrics           | Track proof success rate, average correction iterations, API usage, and latency.             |

---

## 5. System Architecture  

```mermaid
flowchart TD
    A[User Theorem] --> B[SEDD Model]
    B --> C[Noise Injector]
    C --> D[RAG Retriever]
    D --> E[Claude Denoiser]
    E --> F[Formal Checker]
    F --✅ passes--> G[Final Proof]
    F --❌ fails--> H[Retry Logic]
    H --> C
