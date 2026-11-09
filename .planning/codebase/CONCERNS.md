---
last_mapped_commit: c3069a5aeec26cd962835410624faad03b6e8e25
last_mapped_at: 2026-11-09
---
# Technical Concerns & Debt

**Analysis Date:** 2026-11-09

## Current Limitations & Technical Concerns

### 1. Free-Tier API Rate Limits & Daily Quotas
- **Google GenAI Free Tier**: Free tier for `gemini-3.8-flash` enforces a strict 20 requests/day quota (`RESOURCE_EXHAUSTED`). For large evaluation batches, `JUDGE_PROVIDER=mistral` (using `open-mistral-7b`) is configured as the active fallback.
- **Groq Free Tier**: High tokens-per-minute throughput, but sudden burst requests can hit HTTP 429. Pacing (`EVAL_SLEEP_SECONDS=2`) and exponential retry in `rag_pipeline.py` protect against this.

### 2. Windows SQLite File Locking on ChromaDB
- **Issue:** On Windows, ChromaDB's backing SQLite file (`chroma.sqlite3`) remains locked while Python or Streamlit maintains an active handle. Calling `shutil.rmtree()` on the folder results in `[WinError 32]`.
- **Mitigation:** Applied `delete_collection()` in `app.py` rather than filesystem directory deletion.
- **Residual Risk:** If multiple concurrent processes (e.g. Streamlit + a terminal script) access `chroma_db/` simultaneously, SQLite write locks may timeout. Experiments should strictly use isolated directories under `eval_dbs/`.

### 3. Deprecated LangChain Classes
- **Chroma:** LangChain emits a deprecation warning:
  `The class Chroma was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version exists in langchain-chroma.`
  - Migration plan: Upgrade to `langchain-chroma` package in a future refactor wave.

### 4. Space in Project Folder Path on Windows
- **Issue:** The repository folder name contains a space: `2_RAGproject- DocMind`.
- **Impact:** `uv`'s launcher trampolines (e.g. `streamlit.exe`) fail with `uv trampoline failed to canonicalize script path`.
- **Resolution:** Python module invocation syntax (`python -m streamlit run app.py` or `python -m pytest`) must be used instead of calling `.exe` wrappers directly.

### 5. Document Ingestion Encoding Artifacts
- **Issue:** Extracted PDF text contains surrogate UTF-16 characters that cause `UnicodeEncodeError: 'utf-8' codec can't encode character '\ud835'`.
- **Resolution:** Custom `clean_text()` surrogate-pass filter must be maintained across all loaders and ingestion scripts.

---

*Concerns analysis: 2026-11-09*
*Update as technical debt is addressed or new failure modes emerge*
