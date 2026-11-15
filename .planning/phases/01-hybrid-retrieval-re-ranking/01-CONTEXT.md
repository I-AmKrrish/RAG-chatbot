# Phase 1: Hybrid Retrieval & Re-ranking - Context

**Gathered:** 2026-11-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 elevates retrieval quality for DocMind by combining sparse keyword search (BM25) with dense vector search (ChromaDB + MistralAIEmbeddings) via LangChain's `EnsembleRetriever`, followed by a local cross-encoder re-ranking stage (FlashRank). It delivers verifiable improvements in Hit Rate @ k and MRR while keeping existing baseline pipelines operational.
</domain>

<decisions>
## Implementation Decisions

### Hybrid Search & Ensemble Fusion
- **D-01:** Combine BM25 keyword retrieval with ChromaDB dense vector search using LangChain's `EnsembleRetriever` with default balanced weights `[0.5, 0.5]`.
- **D-02:** Build the BM25 index directly from the same document chunk set as ChromaDB to guarantee 1:1 chunk parity, chunk ID alignment, and metadata consistency (source page citation tracking).

### Re-ranking Strategy & Candidate Depth
- **D-03:** Use a local, lightweight cross-encoder model via `FlashRank` (`flashrank` Python package) for candidate re-ranking — ensuring high-precision semantic re-ordering with zero cloud API costs and sub-50ms execution latency.
- **D-04:** Set ensemble candidate fetch depth to `fetch_k=10` (or 2.5x final k), re-ranking all 10 candidates with the cross-encoder and selecting the top `k=4` highest-scoring chunks for prompt assembly.

### Pipeline Architecture & UI Compatibility
- **D-05:** Implement the hybrid re-ranking pipeline within `rag_pipeline.py` in a modular, backward-compatible fashion.
- **D-06:** Expose a retrieval mode selector in the Streamlit sidebar (`app.py`), allowing users to toggle between "Standard MMR" and "Hybrid + Re-ranking" without breaking existing workflows or requiring database re-indexing.

### Verification & Quality Gates
- **D-07:** Benchmark the upgraded retriever using `evaluate_rag.py` against the 30-item evaluation dataset (`eval_dataset.json`, dev split). The upgrade is validated when Hit Rate @ 4 and MRR match or exceed baseline MMR scores.

### Agent Discretion
- Tokenization preprocessing for BM25 (e.g., standard lowercased alphanumeric word tokenization).
- Graceful degradation: if re-ranking fails or encounters empty candidate sets, the pipeline falls back seamlessly to the raw ensemble results without raising exceptions.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Specifications
- `.planning/PROJECT.md` — Core value, constraints, and architecture guidelines
- `.planning/REQUIREMENTS.md` — Requirements RETR-01, RETR-02, RETR-03 definitions
- `.planning/ROADMAP.md` — Phase 1 goals, plans, and success criteria

### Codebase Architecture & Reference Implementations
- `.planning/codebase/ARCHITECTURE.md` — System architecture and retrieval layer specifications
- `.planning/codebase/STACK.md` — Framework versions and runtime environment details
- `rag_pipeline.py` — Baseline RAG pipeline and LLM retry wrapper
- `evaluate_rag.py` — Evaluation benchmark execution engine
- `metrics_deterministic.py` — Mathematical definitions for Hit Rate @ k and MRR
</canonical_refs>

<deferred>
## Deferred Ideas

- Cloud-hosted rerankers (e.g., Cohere Rerank API) — deferred to keep local costs at zero.
- Multi-query expansion and HyDE (Hypothetical Document Embeddings) — logged for v2 milestone.
</deferred>

