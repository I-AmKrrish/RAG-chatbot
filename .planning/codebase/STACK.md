---
last_mapped_commit: c3069a5aeec26cd962835410624faad03b6e8e25
last_mapped_at: 2026-11-09
---
# Technology Stack

**Analysis Date:** 2026-11-09

## Languages

**Primary:**
- Python 3.12 - Core application, vector database ingestion, Streamlit UI, evaluation engine, and test suites.

## Runtime

**Environment:**
- CPython 3.12.1 (Windows x86_64, PowerShell)
- Virtual Environment located at `.venv/` (managed via `uv 0.11.7`)

**Package Manager:**
- `uv` (pip-compatible interface: `uv pip`)
- Requirements manifest: `requirements.txt`

## Frameworks

**Core AI / LLM Orchestration:**
- `langchain` (0.3.30) & `langchain-core` (0.3.86) - RAG chaining, retrieval abstractions, prompt templates
- `langchain-groq` (1.1.3) & `groq` (0.37.1) - Primary inference engine (`ChatGroq`)
- `langchain-mistralai` (1.1.4) & `mistralai` - Vector embedding generator (`MistralAIEmbeddings`) and fallback evaluation judge (`ChatMistralAI`)
- `langchain-google-genai` (4.2.2) & `google-genai` - Primary evaluation judge (`ChatGoogleGenerativeAI`)

**Vector Storage:**
- `chromadb` (1.5.9) & `langchain-community` (0.3.31) - Local embedded vector database persisted in SQLite (`chroma.sqlite3`)

**Document Processing:**
- `pypdf` (6.10.2) & `langchain-community.document_loaders` (`PyPDFLoader`) - PDF extraction
- `langchain-text-splitters` (0.3.11) (`RecursiveCharacterTextSplitter`) - Text chunking with overlap

**Frontend / UI:**
- `streamlit` (1.57.0) - Reactive browser-based chat interface (`app.py`)

**Evaluation & Testing:**
- `ragas` (0.3.1) - LLM-assisted evaluation framework (faithfulness, answer relevancy, context precision, context recall)
- `datasets` (5.1.0) & `pyarrow` (24.0.0) - Dataset manipulation backend for RAGAS
- `pytest` (9.2.0) - Unit test runner for deterministic metrics

## Key Dependencies

**Critical:**
- `langchain_groq` - Powers high-speed chat completion using `openai/gpt-oss-120b`
- `langchain_mistralai` - Generates vector embeddings for ChromaDB (`MistralAIEmbeddings`)
- `chromadb` - Vector index and MMR retrieval engine
- `streamlit` - Interactive multi-turn chat UI and file upload processor
- `ragas` - Metric calculation engine for RAG quality validation

## Configuration

**Environment Configuration:**
- `.env` at project root loaded via `python-dotenv` (`load_dotenv(Path(__file__).parent / ".env")`)
- Required API keys:
  - `GROQ_API_KEY`: Required for `ChatGroq` in `app.py`, `main.py`, and `rag_pipeline.py`
  - `MISTRAL_API_KEY`: Required for `MistralAIEmbeddings` in indexing, retrieval, and judge fallback
  - `GOOGLE_API_KEY`: Required for `ChatGoogleGenerativeAI` in evaluation judging
- Runtime tuning variables:
  - `EVAL_SLEEP_SECONDS`: Rate-limit sleep duration between evaluation questions (default: 2)
  - `CHAT_MODEL`: Chatbot model override (default: `openai/gpt-oss-120b`)
  - `JUDGE_PROVIDER`: Evaluation judge selection (`gemini` or `mistral`, default: `gemini`)
  - `JUDGE_MODEL`: Evaluation judge model override

## Platform Requirements

**Development:**
- Windows 10/11 x64, PowerShell 7 / Windows PowerShell
- Python 3.12 with `.venv` created via `uv`

**Production:**
- Local or containerized Python 3.12 environment with Streamlit port 8501 exposed

---

*Stack analysis: 2026-11-09*
*Update after major dependency changes*
