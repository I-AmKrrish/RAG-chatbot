---
last_mapped_commit: c3069a5aeec26cd962835410624faad03b6e8e25
last_mapped_at: 2026-11-09
---
# Testing Practices & Test Suites

**Analysis Date:** 2026-11-09

## Test Framework

- **Runner:** `pytest` (version 9.2.0)
- **Execution Command:**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```
- **Configuration:** Automatically loads pytest fixtures and plugins (`anyio`, `langsmith`).

## Test Structure

```
tests/
└── test_metrics.py    # Unit tests for deterministic RAG metrics
```

## Unit Test Coverage (`tests/test_metrics.py`)

The test suite validates the pure mathematical scoring functions in `metrics_deterministic.py` without requiring external API calls:

1. **`hit_rate_at_k`**:
   - `test_hit_rate_at_k_normal_hit`: Verifies positive hit when retrieved pages contain any gold page.
   - `test_hit_rate_at_k_normal_miss`: Verifies 0 score when retrieved pages contain no gold pages.
   - `test_hit_rate_at_k_empty_retrieved`: Verifies handling of empty retrieval result.
   - `test_hit_rate_at_k_unanswerable_empty_source`: Verifies that unanswerable items (`source_pages=[]`) return `None` to be safely skipped.

2. **`mrr` (Mean Reciprocal Rank)**:
   - `test_mrr_first_rank`: First retrieved page is correct -> score = 1.0.
   - `test_mrr_second_rank`: Second retrieved page is correct -> score = 0.5.
   - `test_mrr_miss`: No gold pages in retrieval results -> score = 0.0.
   - `test_mrr_empty_lists_and_unanswerable`: Handles empty lists and unanswerables gracefully.

3. **`refusal_detected`**:
   - Validates case-insensitivity on `"could not find the answer"`.
   - Checks positive detection on variations and negative detection on normal answers.
   - Ensures robustness against empty strings and `None`.

4. **`compute_refusal_rates`**:
   - Computes correct refusal rate (refusals on unanswerable items).
   - Computes false refusal rate (refusals on answerable items).
   - Tests edge cases with empty lists.

**Current Test Results:**
- 11 passed in 0.05s (100% pass rate).

## End-to-End Evaluation Testing

In addition to pytest, end-to-end evaluation is tested via:
```powershell
# 5-question smoke test
.\.venv\Scripts\python.exe evaluate_rag.py --label smoke --limit 5

# Hyperparameter sweep verification
.\.venv\Scripts\python.exe run_experiments.py --preset chunking --limit 5 --yes
```

---

*Testing analysis: 2026-11-09*
*Update after adding new test suites or modifying testing patterns*
