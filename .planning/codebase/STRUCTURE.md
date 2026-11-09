---
last_mapped_commit: c3069a5aeec26cd962835410624faad03b6e8e25
last_mapped_at: 2026-11-09
---
# Directory Structure

**Analysis Date:** 2026-11-09

## Project Layout

```
2_RAGproject- DocMind/
├── .env                         # Environment variables (API keys, run configs)
├── .gitignore                    # Version control exclusions (.env, .venv, etc.)
├── requirements.txt             # Python dependency manifest
├── skills-lock.json             # Agent skills lock configuration
│
├── app.py                       # Streamlit web application & UI
├── main.py                      # Interactive terminal RAG chat loop
├── create_database.py           # Offline PDF indexing & Chroma DB builder
├── download_docs.py             # Script for fetching external documentation
├── ingest_docs.py               # Markdown and document parsing utility
│
├── rag_pipeline.py              # Shared RAG pipeline engine with rate-limit retry
├── judge.py                     # Evaluation judge LLM & embeddings factory
├── metrics_deterministic.py     # Pure-function evaluation metrics (Hit Rate, MRR, Refusal)
├── eval_dataset.json            # 30-item ground truth Q&A evaluation benchmark
├── generate_draft_dataset.py    # LLM-assisted draft question generator
├── evaluate_rag.py              # Single-command evaluation & benchmark runner
├── run_experiments.py           # Automated experiment sweeps (chunking, k, search)
│
├── chroma_db/                   # Primary persistent Chroma vector database
│   ├── chroma.sqlite3           # SQLite metadata & vector index table
│   └── <uuid>/                  # Vector binary index segment
│
├── document_loaders/            # Source PDF files
│   ├── deeplearning.pdf         # Primary reference book (Nikhil Buduma)
│   ├── GRU.pdf                  # Gated Recurrent Unit reference PDF
│   └── test.py                  # Loader testing script
│
├── tests/                       # Automated unit tests
│   └── test_metrics.py          # Pytest suite for deterministic metrics
│
├── results/                     # Generated evaluation outputs
│   ├── runs.csv                 # Cumulative run-level benchmark summary
│   ├── <label>_per_question.csv # Detailed per-item evaluation scores
│   └── comparison_<preset>.md   # Formatted markdown comparison tables
│
├── eval_dbs/                    # Isolated experiment vector stores
│   └── cs<size>/                # Vector DB built for specific chunk size
│
├── temp_uploads/                # Staging directory for Streamlit PDF uploads
└── .planning/                   # GSD planning & project intelligence
    └── codebase/                # Codebase knowledge maps (7 documents)
```

## Key Directory Roles

- **Root Directory (`./`)**: Contains executable entry points (`app.py`, `evaluate_rag.py`, `run_experiments.py`), environment configuration, and pipeline definitions.
- **`chroma_db/`**: Production persistent storage for document vectors. Built using `MistralAIEmbeddings` and never modified by evaluation experiments.
- **`eval_dbs/`**: Workspace for isolated experiment vector stores (e.g. `eval_dbs/cs500/`, `eval_dbs/cs1500/`). Prevents contamination of the main database.
- **`document_loaders/`**: Storage for raw source PDF textbooks and loader verification scripts.
- **`results/`**: Output directory for evaluation metrics, detailed per-question CSV logs, and markdown experiment comparison tables.
- **`tests/`**: Unit test suites executed via `pytest`.

## Naming Conventions

- **Python Scripts**: `snake_case.py` (e.g., `rag_pipeline.py`, `create_database.py`)
- **Evaluation Outputs**:
  - Run logs: `results/<label>_per_question.csv`
  - Cumulative summary: `results/runs.csv`
  - Markdown comparison: `results/comparison_<preset>.md`
- **Database Directories**: `chroma_db/` (production), `eval_dbs/cs<chunk_size>/` (experiments)

---

*Structure analysis: 2026-11-09*
*Update after directory reorganizations or new module additions*
