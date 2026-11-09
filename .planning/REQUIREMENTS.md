# Requirements: DocMind PDF RAG Chatbot

**Defined:** 2026-11-10
**Core Value:** Accurate, hallucination-free document answers with verifiable source-page citations and strict refusals for out-of-domain queries.

## v1 Requirements

Requirements for the current development milestone. Each requirement maps to a roadmap phase.

### Advanced Retrieval & Re-ranking

- [ ] **RETR-01**: Implement Hybrid Search combining BM25 sparse keyword retriever with dense vector ChromaDB search via LangChain `EnsembleRetriever`.
- [ ] **RETR-02**: Integrate local cross-encoder re-ranking (FlashRank) to score and filter candidate chunks prior to LLM prompt assembly.
- [ ] **RETR-03**: Verify retrieval quality by measuring Hit Rate @ k and MRR improvements in `evaluate_rag.py` against baseline scores.

### Persistent Conversation Memory

- [ ] **MEM-01**: Implement local SQLite storage schema (`chat_history.db`) for tracking user sessions, conversation threads, timestamps, and message pairs.
- [ ] **MEM-02**: Add multi-turn conversational context to the RAG prompt, enabling contextual follow-up questions while preventing prompt bloat.
- [ ] **MEM-03**: Integrate persistent session selector in Streamlit sidebar, allowing users to resume, view, and clear historical chat threads.

### Multi-Document Knowledge Management

- [ ] **DOC-01**: Extend ingestion to support multi-PDF batch uploads and folder scanning with document metadata tagging.
- [ ] **DOC-02**: Provide document-level filtering in Streamlit sidebar to query individual books or search the entire library simultaneously.

## v2 Requirements

Deferred to future releases.

### Agentic Capabilities & Streaming

- **AGENT-01**: Query rewriting and HyDE (Hypothetical Document Embeddings) for complex ambiguous questions.
- **STREAM-01**: Real-time token streaming in Streamlit UI using LangChain callback handlers for immediate user feedback.
- **OBSERV-01**: Telemetry and trace logging via Langfuse or LangSmith for production monitoring.

## Out of Scope

Explicitly excluded to maintain project boundaries.

| Feature | Reason |
|---------|--------|
| Paid Cloud Rerankers (e.g. Cohere Rerank API) | User selected zero-cost local cross-encoder (FlashRank) |
| Groq-based Evaluation Judging | Judge model must be a different model family (Gemini / Mistral) to prevent self-grading bias |
| Direct `shutil.rmtree` on Active ChromaDB | Windows SQLite mandatory file locks cause crashes; in-memory collection clearing is enforced |

## Traceability

Which phases cover which requirements.

| Requirement | Phase | Status |
|-------------|-------|--------|
| RETR-01 | Phase 1: Hybrid Retrieval & Re-ranking | Pending |
| RETR-02 | Phase 1: Hybrid Retrieval & Re-ranking | Pending |
| RETR-03 | Phase 1: Hybrid Retrieval & Re-ranking | Pending |
| MEM-01 | Phase 2: Persistent Conversation Memory | Pending |
| MEM-02 | Phase 2: Persistent Conversation Memory | Pending |
| MEM-03 | Phase 2: Persistent Conversation Memory | Pending |
| DOC-01 | Phase 3: Multi-Document Knowledge Management | Pending |
| DOC-02 | Phase 3: Multi-Document Knowledge Management | Pending |

**Coverage:**
- v1 requirements: 8 total
- Mapped to phases: 8
- Unmapped: 0 ✓

---
*Requirements defined: 2026-11-10*
*Last updated: 2026-11-10 after initialization*
