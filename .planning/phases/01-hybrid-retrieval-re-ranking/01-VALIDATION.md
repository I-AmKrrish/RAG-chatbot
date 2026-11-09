---
phase: "01"
slug: "hybrid-retrieval-re-ranking"
status: ready
nyquist_compliant: true
wave_0_complete: false
created: "2026-11-10"
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.2.0 |
| **Config file** | `pytest.ini` (Wave 0 adds `pythonpath = .` and `testpaths = tests`) |
| **Quick run command** | `.venv\Scripts\pytest.exe tests/` |
| **Full suite command** | `.venv\Scripts\pytest.exe tests/ -v` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv\Scripts\pytest.exe tests/`
- **After every plan wave:** Run `.venv\Scripts\pytest.exe tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green + benchmark logged to `results/runs.csv`
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | Fails When | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|------------|--------|
| 01-01-01 | 01 | 1 | RETR-01, RETR-02 | T-01-SC | Pinned dependency versions and safe import paths | tracer | `.venv\Scripts\pytest.exe tests/test_retrieval.py -k test_hybrid_retriever_tracer_path -v` | non-zero exit code or "FAILED" in test output | ⬜ pending |
| 01-01-02 | 01 | 1 | RETR-01 | T-01-01 | Safe BM25 query handling and 1:1 chunk parity | unit | `.venv\Scripts\pytest.exe tests/test_retrieval.py -k "test_bm25 or test_ensemble" -v` | non-zero exit code or "FAILED" in test output | ⬜ pending |
| 01-01-03 | 01 | 1 | RETR-02 | T-01-02 | Candidate depth capped at 10 to prevent CPU DoS + graceful fallback | unit | `.venv\Scripts\pytest.exe tests/test_retrieval.py -v` | non-zero exit code or "FAILED" in test output | ⬜ pending |
| 01-02-01 | 02 | 2 | RETR-01, RETR-02 | T-01-03 | Strict prompt framing preserved across retrieval modes | unit / int | `.venv\Scripts\pytest.exe tests/test_retrieval.py -k test_pipeline_hybrid_mode -v` | non-zero exit code or "FAILED" in test output | ⬜ pending |
| 01-02-02 | 02 | 2 | RETR-03 | T-01-04 | Rate limit throttling via EVAL_SLEEP_SECONDS | integration | `.venv\Scripts\python.exe evaluate_rag.py --label hybrid_eval_smoke --split dev --limit 5 --judge-provider mistral --search-type hybrid` | non-zero exit code, "ERROR", or missing runs.csv row | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pytest.ini` — Root pytest configuration setting `pythonpath = .` and `testpaths = tests`
- [ ] `tests/test_retrieval.py` — Stubs and fixtures for BM25, Ensemble, and FlashRank testing
- [ ] `requirements.txt` — Add `rank-bm25`, `flashrank`, and `"huggingface-hub>=2.4.0"`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Streamlit Sidebar Retrieval Mode Toggle | RETR-01, RETR-02 | Interactive UI control | Run `streamlit run app.py`, verify sidebar radio displays "Standard MMR" and "Hybrid + Re-ranking" and toggles without crashing. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready


