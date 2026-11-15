import pytest
from metrics_deterministic import (
    hit_rate_at_k,
    mrr,
    refusal_detected,
    compute_refusal_rates,
)


def test_hit_rate_at_k_normal_hit():
    source_pages = [5, 12]
    retrieved_pages = [1, 5, 20, 22]
    assert hit_rate_at_k(source_pages, retrieved_pages) == 1


def test_hit_rate_at_k_normal_miss():
    source_pages = [5, 12]
    retrieved_pages = [1, 2, 3, 4]
    assert hit_rate_at_k(source_pages, retrieved_pages) == 0


def test_hit_rate_at_k_empty_retrieved():
    source_pages = [5]
    retrieved_pages = []
    assert hit_rate_at_k(source_pages, retrieved_pages) == 0


def test_hit_rate_at_k_unanswerable_empty_source():
    # Unanswerable items have empty source_pages
    assert hit_rate_at_k([], [1, 2, 3]) is None
    assert hit_rate_at_k([], []) is None


def test_mrr_first_rank():
    source_pages = [10]
    retrieved_in_order = [10, 2, 3, 4]
    assert mrr(source_pages, retrieved_in_order) == pytest.approx(1.0)


def test_mrr_second_rank():
    source_pages = [15, 20]
    retrieved_in_order = [2, 15, 8, 20]
    assert mrr(source_pages, retrieved_in_order) == pytest.approx(0.5)


def test_mrr_miss():
    source_pages = [10]
    retrieved_in_order = [1, 2, 3, 4]
    assert mrr(source_pages, retrieved_in_order) == 0.0


def test_mrr_empty_lists_and_unanswerable():
    # Empty retrieved
    assert mrr([5], []) == 0.0
    # Unanswerable (empty source pages)
    assert mrr([], [1, 2, 3]) is None
    assert mrr([], []) is None


def test_refusal_detected_cases():
    assert refusal_detected("I could not find the answer in the given documents.") is True
    assert refusal_detected("I COULD NOT FIND THE ANSWER in the given documents") is True
    assert refusal_detected("Sorry, I could not find the answer.") is True
    assert refusal_detected("Deep learning is a subset of machine learning.") is False
    assert refusal_detected("") is False
    assert refusal_detected(None) is False


def test_compute_refusal_rates():
    items = [
        # Unanswerable items (correct behavior is refusal)
        {"source_pages": [], "type": "unanswerable", "answer": "I could not find the answer in the given documents."},
        {"source_pages": [], "type": "unanswerable", "answer": "The answer is 42."},
        # Answerable items (refusal is a false refusal)
        {"source_pages": [1], "type": "simple", "answer": "Deep learning uses neural networks."},
        {"source_pages": [2], "type": "multi_chunk", "answer": "I could not find the answer."},
    ]
    rates = compute_refusal_rates(items)
    # Unanswerable: 1 correct refusal out of 2 -> 0.5
    assert rates["correct_refusal_rate"] == 0.5
    # Answerable: 1 false refusal out of 2 -> 0.5
    assert rates["false_refusal_rate"] == 0.5
    assert rates["n_unanswerable"] == 2
    assert rates["n_answerable"] == 2


def test_compute_refusal_rates_empty():
    rates = compute_refusal_rates([])
    assert rates["correct_refusal_rate"] is None
    assert rates["false_refusal_rate"] is None
    assert rates["n_unanswerable"] == 0
    assert rates["n_answerable"] == 0

