# Roadmap: DocMind PDF RAG Chatbot

## Overview

This roadmap advances DocMind from a single-document RAG prototype into an advanced, production-grade document intelligence system. Milestone 1 delivers three structured phases: (1) Hybrid Retrieval and local Cross-Encoder Re-ranking to maximize answer grounding; (2) Persistent SQLite conversation memory for multi-turn conversational threads in Streamlit; and (3) Multi-document knowledge base management with metadata filtering.

## Phases

- [ ] **Phase 1: Hybrid Retrieval & Re-ranking** - Combine BM25 keyword matching with ChromaDB vector search and local cross-encoder re-ranking.
- [ ] **Phase 2: Persistent Conversation Memory** - Store multi-turn conversation sessions in SQLite with Streamlit thread management.
- [ ] **Phase 3: Multi-Document Knowledge Management** - Enable multi-PDF batch ingestion, catalog management, and document-level filtering.

## Phase Details

### Phase 1: Hybrid Retrieval & Re-ranking
**Goal**: Elevate retrieval precision by fusing dense semantic embeddings with sparse keyword search and re-ranking candidate chunks locally before generation.
**Depends on**: Nothing (builds upon existing ChromaDB and pipeline)
**Requirements**: RETR-01, RETR-02, RETR-03
**Success Criteria** (what must be TRUE):
  1. LangChain `EnsembleRetriever` merges BM25 keyword retrieval with ChromaDB dense vector search with configurable weights.
  2. Local cross-encoder (FlashRank) re-scores top candidate chunks without external API latency or costs.
  3. Running `evaluate_rag.py` shows measurable improvement in Hit Rate @ k and MRR over baseline.
**Plans**: 2 plans

Plans:
- [ ] 01-01: Implement BM25 retriever indexer, local FlashRank cross-encoder reranker, and ensemble pipeline.
- [ ] 01-02: Integrate hybrid retrieval into `rag_pipeline.py` and run benchmark verification in `evaluate_rag.py`.

### Phase 2: Persistent Conversation Memory
**Goal**: Provide full multi-turn conversational memory persisted locally in SQLite, allowing users to resume past sessions.
**Depends on**: Phase 1
**Requirements**: MEM-01, MEM-02, MEM-03
**Success Criteria** (what must be TRUE):
  1. SQLite database (`chat_history.db`) stores sessions, conversation threads, message timestamps, and source chunk metadata.
  2. Streamlit sidebar enables creating new chats, browsing historical threads, and loading previous discussions.
  3. RAG pipeline injects recent conversational context to accurately resolve follow-up questions (e.g. "tell me more about that").
**Plans**: 2 plans

Plans:
- [ ] 02-01: Create SQLite session store, thread schema, and history query abstractions.
- [ ] 02-02: Connect Streamlit UI to SQLite session manager and integrate conversational buffer into the RAG chain.

### Phase 3: Multi-Document Knowledge Management
**Goal**: Transform the system into a multi-document library supporting batch PDF indexing and scoped search.
**Depends on**: Phase 2
**Requirements**: DOC-01, DOC-02
**Success Criteria** (what must be TRUE):
  1. Batch ingestion reads all PDFs in a designated directory or multi-upload queue with unique `doc_id` metadata.
  2. Streamlit sidebar provides document catalog selector to search across "All Documents" or filter to specific PDFs.
**Plans**: 2 plans

Plans:
- [ ] 03-01: Extend ingestion pipeline to support multi-PDF batching and document metadata schemas.
- [ ] 03-02: Update Streamlit interface with document library panel and metadata-filtered retrieval queries.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Hybrid Retrieval & Re-ranking | 0/2 | Not started | - |
| 2. Persistent Conversation Memory | 0/2 | Not started | - |
| 3. Multi-Document Knowledge Management | 0/2 | Not started | - |

---
*Roadmap defined: 2026-11-10*
*Last updated: 2026-11-10 after initialization*
