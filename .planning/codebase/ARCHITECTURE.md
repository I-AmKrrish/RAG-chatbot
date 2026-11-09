---
last_mapped_commit: c3069a5aeec26cd962835410624faad03b6e8e25
last_mapped_at: 2026-11-09
---
# System Architecture

**Analysis Date:** 2026-11-09

## Architectural Pattern

DocMind follows a decoupled Retrieval-Augmented Generation (RAG) architecture consisting of four core subsystems:
1. **Ingestion & Indexing Pipeline**: PDF extraction, surrogate-safe text cleaning, recursive character chunking, Mistral vector embedding, and ChromaDB persistence.
2. **Retrieval & Chaining Layer**: Maximal Marginal Relevance (MMR) vector similarity search with rank-ordered source tracking.
3. **Application Interfaces**: Multi-turn Streamlit web interface (`app.py`) and command-line conversational runner (`main.py`).
4. **Evaluation & Experimentation Framework**: Pure deterministic metrics, RAGAS multi-metric judging with provider separation, and preset hyperparameter experiments.

```
                      +-----------------------------+
                      |   PDF Document Ingestion    |
                      |  (PyPDFLoader + clean_text) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | RecursiveCharacterSplitter  |
                      |   (size=1000, overlap=100)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    MistralAIEmbeddings      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |          ChromaDB           |
                      |  (chroma.sqlite3 / MMR k=4) |
                      +--------------+--------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
      +---------------------+                 +---------------------+
      |   Streamlit App UI  |                 | RAG Pipeline Engine |
      |      (app.py)       |                 |  (rag_pipeline.py)  |
      +----------+----------+                 +----------+----------+
                 |                                       |
                 v                                       v
      +---------------------+                 +---------------------+
      |   ChatGroq (LLM)    |                 |  Evaluation Suite   |
      | (openai/gpt-oss-120b|                 |  (evaluate_rag.py)  |
      +---------------------+                 +----------+----------+
                                                         |
                                                         v
                                              +---------------------+
                                              | Independent Judges  |
                                              | (Gemini / Mistral)  |
                                              +---------------------+
```

## Layers & Components

### 1. Ingestion Layer
- **`create_database.py`**: Reads `document_loaders/deeplearning.pdf`, cleans surrogate strings, splits documents into 1000-character chunks with 100-character overlap, generates embeddings via Mistral AI, and initializes `./chroma_db`.
- **`app.py` Upload Pipeline**: Accepts dynamic PDF drag-and-drop uploads via `st.file_uploader`, saves to `temp_uploads/`, indexes vectors into Chroma, safely deletes existing collections, and sets session readiness.

### 2. Retrieval Layer
- **Vector Search Strategy**: Maximal Marginal Relevance (`search_type="mmr"`)
- **Default Hyperparameters**:
  - `k`: 4 (number of final context chunks returned to prompt)
  - `fetch_k`: 10 (candidate pool retrieved from vector space before diversity filtering)
  - `lambda_mult`: 0.5 (balance between relevance and diversity)
- **Metadata Preservation**: Retains 1-indexed document page numbers (`doc.metadata["page"] + 1`) to enable citation tracking.

### 3. Generation Layer
- **Chat Model**: `ChatGroq(model="openai/gpt-oss-120b", temperature=0)`
- **Prompt Architecture**: Strict zero-shot grounding prompt:
  - System message forces reliance *only* on the provided context chunks.
  - Mandated fallback text on missing information: `"I could not find the answer in the given documents"`.
- **Fault Tolerance**: `rag_pipeline.py` implements exponential backoff retry for HTTP 429 rate limits, with graceful degradation returning an empty answer and error code rather than raising unhandled exceptions.

### 4. Evaluation & Benchmarking Layer
- **`metrics_deterministic.py`**: Non-LLM mathematical evaluations:
  - Hit Rate @ k (page retrieval verification)
  - Mean Reciprocal Rank (MRR)
  - Refusal Detection and classification (correct refusal vs false refusal)
- **`judge.py`**: Separates evaluation LLM family from the chatbot model:
  - Primary: Google Gemini (`gemini-3.8-flash`)
  - Fallback: Mistral AI (`open-mistral-7b`)
- **`evaluate_rag.py`**: Orchestrates single-sample RAGAS execution, logs detailed diagnostics to `results/<label>_per_question.csv`, and appends aggregate runs to `results/runs.csv`.
- **`run_experiments.py`**: Runs controlled hyperparameter sweeps across chunk sizes, k values, and retrieval algorithms in isolated directories (`eval_dbs/`).

## Entry Points

1. **`app.py`**: Interactive Web UI (`python -m streamlit run app.py`).
2. **`main.py`**: Terminal interactive loop (`python main.py`).
3. **`create_database.py`**: Knowledge base builder (`python create_database.py`).
4. **`evaluate_rag.py`**: Evaluation CLI (`python evaluate_rag.py --label <name>`).
5. **`run_experiments.py`**: Automated experiment runner (`python run_experiments.py --preset <preset>`).

---

*Architecture analysis: 2026-11-09*
*Update after structural or architectural changes*
