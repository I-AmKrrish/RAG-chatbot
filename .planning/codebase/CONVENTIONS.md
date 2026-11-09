---
last_mapped_commit: c3069a5aeec26cd962835410624faad03b6e8e25
last_mapped_at: 2026-11-09
---
# Coding Conventions & Patterns

**Analysis Date:** 2026-11-09

## Code Style & Formatting

- **Language Standard:** Python 3.12 syntax with type annotations where appropriate (`typing.List`, `typing.Optional`, `typing.Dict`).
- **File Encoding:** UTF-8 throughout.
- **Path Resolution:** Standardized on `pathlib.Path` for filesystem navigation and environment file loading:
  ```python
  from pathlib import Path
  from dotenv import load_dotenv
  load_dotenv(Path(__file__).parent / ".env")
  ```

## Architectural Conventions

### 1. Secret Protection & Environment Isolation
- API keys are never printed, logged, or hardcoded.
- Credentials (`GROQ_API_KEY`, `MISTRAL_API_KEY`, `GOOGLE_API_KEY`) are accessed strictly via `os.getenv()`.
- Scripts fail fast with descriptive error messages if required keys are missing, without dumping stack traces exposing environment details.

### 2. Rate Limiting & Backoff
- All LLM invocations in evaluation and pipelines must be resilient to HTTP 429 / quota exhaustion.
- Implemented pattern (`rag_pipeline.py`):
  - Exponential backoff starting at 2.0s with factor 2.0.
  - Inspection of `Retry-After` headers if provided by upstream APIs.
  - Pacing delay (`EVAL_SLEEP_SECONDS`, default 2s) between batch queries to prevent provider throttling.

### 3. Graceful Error Handling
- Batch scripts (e.g. `evaluate_rag.py`) must never crash the entire run on a single malformed question.
- Evaluation items with errors return empty answers and log an `"error"` diagnostic string, recorded as `n_errors` in aggregate statistics.

### 4. Text Cleaning for Vector Indexing
- Source PDF text from `pypdf` often contains surrogate pairs that fail UTF-8 encoding on Windows:
  ```python
  def clean_text(text: str) -> str:
      if not text:
          return ""
      return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
  ```
- This normalization is applied in `create_database.py`, `app.py`, and `run_experiments.py` before chunk splitting.

### 5. Windows SQLite File Lock Mitigation
- Windows locks open SQLite database files.
- Never call `shutil.rmtree()` on an active Chroma directory without first dropping collection handles:
  ```python
  if os.path.exists(db_directory):
      try:
          existing_vs = Chroma(persist_directory=db_directory, embedding_function=embedding_model)
          existing_vs.delete_collection()
      except Exception:
          pass
  ```

## Testing Patterns

- Unit tests live under `tests/test_*.py`.
- Pure functions are decoupled from network and external I/O for deterministic testing.
- Unit assertions use `pytest` and `pytest.approx` for floating-point comparisons.

---

*Conventions analysis: 2026-11-09*
*Update after establishing new team patterns or refactoring standards*
