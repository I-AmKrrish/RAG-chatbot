"""
Deterministic, pure-function evaluation metrics for RAG retrieval and response validation.
Contains no API or LLM calls.
"""
from typing import List, Optional, Dict, Any


def hit_rate_at_k(source_pages: List[int], retrieved_pages: List[int]) -> Optional[int]:
    """
    Returns 1 if any gold source page appears in retrieved_pages, else 0.
    For unanswerable items (empty source_pages), returns None so they can be skipped.

    Note: page-based hit rate is approximate because a gold page may span chunks.
    """
    if not source_pages:
        return None
    source_set = set(source_pages)
    return 1 if any(p in source_set for p in retrieved_pages) else 0


def mrr(source_pages: List[int], retrieved_pages_in_rank_order: List[int]) -> Optional[float]:
    """
    Computes Mean Reciprocal Rank (MRR) for the first relevant retrieved page.
    Returns 1.0 / rank (1-based), or 0.0 if no gold page is retrieved.
    For unanswerable items (empty source_pages), returns None.
    """
    if not source_pages:
        return None
    source_set = set(source_pages)
    for rank, p in enumerate(retrieved_pages_in_rank_order, start=1):
        if p in source_set:
            return 1.0 / rank
    return 0.0


def refusal_detected(answer: str) -> bool:
    """
    Returns True if the answer contains 'could not find the answer' (case-insensitive).
    """
    if not answer:
        return False
    return "could not find the answer" in answer.lower()


def compute_refusal_rates(evaluated_items: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Computes correct refusal rate and false refusal rate across evaluated items.

    - Unanswerable items (type == 'unanswerable' or empty source_pages):
      correct behavior is refusal.
    - Answerable items:
      refusal is a false refusal.
    """
    unanswerable_total = 0
    unanswerable_refusals = 0

    answerable_total = 0
    answerable_false_refusals = 0

    for item in evaluated_items:
        source_pages = item.get("source_pages") or []
        is_unanswerable = (item.get("type") == "unanswerable") or (len(source_pages) == 0)
        answer = item.get("answer", "")
        refused = refusal_detected(answer)

        if is_unanswerable:
            unanswerable_total += 1
            if refused:
                unanswerable_refusals += 1
        else:
            answerable_total += 1
            if refused:
                answerable_false_refusals += 1

    correct_refusal_rate = (
        (unanswerable_refusals / unanswerable_total) if unanswerable_total > 0 else None
    )
    false_refusal_rate = (
        (answerable_false_refusals / answerable_total) if answerable_total > 0 else None
    )

    return {
        "correct_refusal_rate": round(correct_refusal_rate, 4) if correct_refusal_rate is not None else None,
        "false_refusal_rate": round(false_refusal_rate, 4) if false_refusal_rate is not None else None,
        "n_unanswerable": unanswerable_total,
        "n_answerable": answerable_total,
    }

