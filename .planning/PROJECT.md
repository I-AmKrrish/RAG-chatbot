# DocMind: PDF RAG Chatbot

## What This Is

DocMind is a high-performance Retrieval-Augmented Generation (RAG) assistant for querying and analyzing PDF documents with grounded answers and source citations. It features a reactive Streamlit web interface, a robust offline ChromaDB indexing pipeline, high-speed LLM inference powered by Groq (`openai/gpt-oss-120b`), and an automated RAGAS evaluation harness.

## Core Value

Accurate, hallucination-free document answers with verifiable source-page citations and strict refusals for out-of-domain queries.

## Requirements

### Validated

- ✓ **INGEST-01**: PDF document ingestion and surrogate-safe text cleaning via PyPDFLoader — existing
- ✓ **SPLIT-01**: Recursive character chunking (1000 size, 100 overlap) — existing
- ✓ **EMBED-01**: Dense vector representations via MistralAIEmbeddings — existing
- ✓ **STORE-01**: Persistent ChromaDB vector database with MMR retrieval (k=4, fetch_k=10, lambda=0.5) — existing
- ✓ **CHAT-01**: Fast LLM generation via ChatGroq with `openai/gpt-oss-120b` and temperature 0 — existing
- ✓ **UI-01**: Reactive Streamlit chat interface with expandable citation source chunks — existing
- ✓ **EVAL-01**: Automated evaluation suite with RAGAS metrics (faithfulness, relevancy, precision, recall) — existing
- ✓ **METRIC-01**: Deterministic retrieval metrics (Hit Rate @ k, MRR, Refusal accuracy) with 100% pytest coverage — existing
- ✓ **EXP-01**: Hyperparameter experiment runner with preset sweeps in isolated directories — existing

### Active

- [ ] **RETR-01**: Implement Hybrid Search combining BM25 keyword search with dense vector retrieval using an ensemble ranker
- [ ] **RETR-02**: Integrate local cross-encoder re-ranking (e.g. FlashRank) for high-precision context filtering before LLM generation
- [ ] **MEM-01**: Create persistent SQLite database schema (`chat_history.db`) for storing chat sessions, threads, and message history
- [ ] **UI-02**: Streamlit session integration to load, resume, and switch between past conversation threads
- [ ] **DOC-01**: Multi-document folder ingestion with per-document filtering options in the UI

### Out of Scope

- Cloud-hosted paid reranking APIs (e.g. Cohere Rerank) — prioritized local cross-encoder with zero API cost
- Groq-based evaluation judge — evaluations require an independent model family (Gemini / Mistral) to prevent self-grading bias
- Direct filesystem directory deletion on active ChromaDB — Windows SQLite file locking causes crashes; collection-level clearing is mandated

## Context

- **Environment**: Windows 11 with PowerShell, virtual environment `.venv` managed with `uv`.
- **Active Document Knowledge Base**: 291 pages (712 chunks) from *Fundamentals of Deep Learning* (Nikhil Buduma) indexed in `./chroma_db`.
- **Rate-Limit Resilience**: Exponential retry and pacing implemented across all LLM inference points.

## Constraints

- **API Limits**: Google Gemini free tier has strict daily quota limits; Mistral AI is maintained as the primary fallback judge.
- **Windows Pathing**: Project directory contains a space (`2_RAGproject- DocMind`); all tools and scripts run via `python -m <module>`.
- **Database Isolation**: Experiments run strictly against `./eval_dbs/` without mutating production `./chroma_db/`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Groq `openai/gpt-oss-120b` for chat | Provides high-capacity 120B parameter reasoning with sub-second inference speeds | ✓ Good |
| MistralAIEmbeddings for vectors | High quality semantic embeddings compatible with Chroma vector space | ✓ Good |
| Local FlashRank cross-encoder | Maximizes retrieval precision without adding API costs or cloud network latency | — Pending |
| Local SQLite for conversation history | Reliable local persistence for chat sessions that survives server reboots | — Pending |
| Separate judge models (Gemini / Mistral) | Eliminates model bias when evaluating Groq chatbot responses | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-11-10 after initialization*
