---
phase: "01"
slug: "hybrid-retrieval-re-ranking"
status: draft
nyquist_compliant: false
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
| **Config file** | `pytest.ini` (Wave 0 adds `pythonpath = .`) |
| **Quick run command** | `python -m pytest tests/` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green + benchmark logged to `results/runs.csv`
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | RETR-01 | — | N/A | config | `python -m pytest --version` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | RETR-01 | T-01-01 | Safe BM25 query handling without regex crashes | unit | `python -m pytest tests/test_retrieval.py::test_bm25_retriever -v` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | RETR-01 | — | N/A | unit | `python -m pytest tests/test_retrieval.py::test_ensemble_retriever -v` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 1 | RETR-02 | T-01-02 | Candidate depth capped at 10 to prevent CPU DoS | unit | `python -m pytest tests/test_retrieval.py::test_flashrank_reranker -v` | ❌ W0 | ⬜ pending |
| 01-01-05 | 01 | 1 | RETR-02 | — | Graceful fallback on empty/failed rerank | unit | `python -m pytest tests/test_retrieval.py::test_reranker_fallback -v` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 2 | RETR-01, RETR-02 | T-01-03 | Strict prompt framing preserved across retrieval modes | unit / int | `python -m pytest tests/test_retrieval.py::test_pipeline_hybrid_mode -v` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 2 | RETR-03 | — | N/A | integration | `python evaluate_rag.py --label test_hybrid --split dev --limit 5 --judge-provider mistral` | ✅ Exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pytest.ini` — Root pytest configuration setting `pythonpath = .`
- [ ] `tests/test_retrieval.py` — Stubs and fixtures for BM25, Ensemble, and FlashRank testing
- [ ] `requirements.txt` — Add `rank-bm25`, `flashrank`, and `"huggingface-hub>=2.4.0"`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Streamlit Sidebar Retrieval Mode Toggle | RETR-01, RETR-02 | Interactive UI control | Run `streamlit run app.py`, verify sidebar radio/select displays "Standard MMR" and "Hybrid + Re-ranking" and toggles without crashing. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
