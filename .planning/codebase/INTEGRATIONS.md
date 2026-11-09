---
last_mapped_commit: c3069a5aeec26cd962835410624faad03b6e8e25
last_mapped_at: 2026-11-09
---
# External Integrations

**Analysis Date:** 2026-11-09

## Overview

DocMind / RAGproject is an end-to-end Retrieval-Augmented Generation (RAG) system interfacing with cloud model APIs, an embedded local vector database, and local document sources.

## Model APIs & LLM Providers

### 1. Groq Cloud API
- **Client Class:** `langchain_groq.ChatGroq` (`groq.Groq`)
- **Authentication:** `GROQ_API_KEY` loaded from `.env`
- **Active Model:** `openai/gpt-oss-120b` (fallback options: `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`)
- **Usage Locations:**
  - `app.py` - User-facing Streamlit chat interface
  - `rag_pipeline.py` - Evaluation inference execution
- **Rate Limits & Handling:** Rate limits handled via exponential backoff in `rag_pipeline.py` (`_call_llm_with_retry`) supporting `Retry-After` headers.

### 2. Mistral AI API
- **Client Class:** `langchain_mistralai.MistralAIEmbeddings`, `langchain_mistralai.ChatMistralAI`
- **Authentication:** `MISTRAL_API_KEY` loaded from `.env`
- **Active Endpoints:**
  - Embeddings: `MistralAIEmbeddings` (powers all vector collections in `./chroma_db` and `./eval_dbs/`)
  - Chat: `open-mistral-7b` (configured in `judge.py` as secondary evaluation judge)
- **Usage Locations:**
  - `create_database.py` - Offline PDF embedding generation
  - `app.py` - Dynamic PDF upload vectorization
  - `rag_pipeline.py` - Retrieval vector query embedding
  - `judge.py` - Evaluation judging & RAGAS wrapper

### 3. Google GenAI (Gemini) API
- **Client Class:** `langchain_google_genai.ChatGoogleGenerativeAI`
- **Authentication:** `GOOGLE_API_KEY` loaded from `.env`
- **Active Model:** `gemini-3.8-flash` (or `gemini-flash-latest`)
- **Usage Locations:**
  - `judge.py` - Primary automated evaluation judge for RAGAS metrics
  - `generate_draft_dataset.py` - Automated candidate question drafting

## Data Storage & Vector Databases

### ChromaDB
- **Type:** Local embedded vector store backed by SQLite
- **Directory Paths:**
  - `./chroma_db/` - Primary production knowledge base (holds 712 chunks from `deeplearning.pdf`)
  - `./eval_dbs/cs<size>/` - Isolated experiment vector stores for hyperparameter evaluation (e.g. `cs500`, `cs1000`, `cs1500`)
- **Concurrency & Locking Behavior:**
  - SQLite maintains mandatory file locks on Windows (`chroma.sqlite3`).
  - Safe in-memory collection dropping (`existing_vs.delete_collection()`) is implemented in `app.py` to prevent `[WinError 32]` file access conflicts.

## Document Ingestion Sources

### PDF Loaders
- **Library:** `langchain_community.document_loaders.PyPDFLoader` (`pypdf`)
- **Input Directories:**
  - `document_loaders/` (contains `deeplearning.pdf`, `GRU.pdf`)
  - `temp_uploads/` (temporary staging directory for user uploads via Streamlit)
- **Preprocessing:** Custom surrogatepass cleaner (`clean_text`) to strip surrogate UTF-16 characters before chunking.

---

*Integrations analysis: 2026-11-09*
*Update after adding or changing external integrations*
