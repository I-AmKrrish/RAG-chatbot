import os
import sys
import time
import json
import csv
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Safe environment loading
load_dotenv(Path(__file__).parent / ".env")

from rag_pipeline import build_pipeline
from judge import get_judge_llm, get_judge_embeddings, get_judge_provider_and_model
from metrics_deterministic import (
    hit_rate_at_k,
    mrr,
    refusal_detected,
    compute_refusal_rates,
)

# RAGAS imports
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


async def score_sample_ragas(sample: SingleTurnSample, is_unanswerable: bool, metrics_dict: dict):
    """
    Score a single sample using RAGAS metrics.
    Skips context_precision and context_recall for unanswerable items.
    """
    scores = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }

    try:
        # Faithfulness
        scores["faithfulness"] = await metrics_dict["faithfulness"].single_turn_ascore(sample)
    except Exception as e:
        scores["faithfulness"] = float("nan")

    try:
        # Answer Relevancy
        scores["answer_relevancy"] = await metrics_dict["answer_relevancy"].single_turn_ascore(sample)
    except Exception as e:
        scores["answer_relevancy"] = float("nan")

    if not is_unanswerable:
        try:
            scores["context_precision"] = await metrics_dict["context_precision"].single_turn_ascore(sample)
        except Exception as e:
            scores["context_precision"] = float("nan")

        try:
            scores["context_recall"] = await metrics_dict["context_recall"].single_turn_ascore(sample)
        except Exception as e:
            scores["context_recall"] = float("nan")

    return scores


def run_evaluation(
    label: str,
    split: str = "dev",
    limit: int = None,
    persist_dir: str = "chroma_db",
    dataset_path: str = "eval_dataset.json",
    judge_provider: str = None,
    k: int = 4,
    search_type: str = "mmr",
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
    chunk_size: int = 1000,
):
    # Rule 2 / Acceptance 3: Validate Keys
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    prov, judge_model_name = get_judge_provider_and_model(provider=judge_provider)
    if prov == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is not set in environment or .env file (required when JUDGE_PROVIDER=gemini).")
        sys.exit(1)
    elif prov == "mistral" and not os.getenv("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY is not set in environment or .env file (required when JUDGE_PROVIDER=mistral).")
        sys.exit(1)

    chat_model_name = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")

    # Print startup configuration
    print("=" * 60)
    print("RAG EVALUATION STARTUP")
    print(f"  Run Label:      {label}")
    print(f"  Split:          {split}")
    print(f"  Chat Model:     {chat_model_name}")
    print(f"  Judge Provider: {prov}")
    print(f"  Judge Model:    {judge_model_name}")
    print(f"  Persist Dir:    {persist_dir}")
    print("=" * 60)

    # 1. Load Dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    # Filter items: IGNORE draft items and count how many skipped
    draft_count = sum(1 for it in all_items if it.get("status") == "draft")
    print(f"Skipping {draft_count} unreviewed items with status 'draft'.")

    reviewed_items = [it for it in all_items if it.get("status") == "reviewed"]

    if split != "all":
        items = [it for it in reviewed_items if it.get("split") == split]
    else:
        items = reviewed_items

    if limit is not None and limit > 0:
        items = items[:limit]

    print(f"Loaded {len(items)} questions for evaluation (split='{split}', limit={limit}).")
    if not items:
        print("No items to evaluate.")
        return {}

    # 2. Build Pipeline
    pipeline = build_pipeline(
        persist_dir=persist_dir,
        k=k,
        search_type=search_type,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        model=chat_model_name
    )

    # 3. Setup Judge
    judge_llm = get_judge_llm(provider=prov, model=judge_model_name)
    judge_embeddings = get_judge_embeddings()
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embed = LangchainEmbeddingsWrapper(judge_embeddings)

    metrics_dict = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    for m in metrics_dict.values():
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embed

    sleep_seconds = float(os.getenv("EVAL_SLEEP_SECONDS", "2"))

    # 4. Evaluate each question
    results_rows = []
    n_errors = 0

    for idx, item in enumerate(items, start=1):
        q_id = item.get("id", f"q{idx:03d}")
        q_text = item["question"]
        gold_answer = item["ground_truth"]
        gold_pages = item.get("source_pages", [])
        q_type = item.get("type", "simple")
        is_unanswerable = (q_type == "unanswerable") or (len(gold_pages) == 0)

        print(f"[{idx}/{len(items)}] Answering {q_id}: {q_text[:60]}...")
        ans_dict = pipeline.answer(q_text)

        answer = ans_dict.get("answer", "")
        contexts = ans_dict.get("contexts", [])
        retrieved_pages = ans_dict.get("source_pages", [])
        latency = ans_dict.get("latency_seconds", 0.0)
        err = ans_dict.get("error")

        if err:
            n_errors += 1
            print(f"   -> Pipeline error: {err}")

        # Deterministic metrics
        hr = hit_rate_at_k(gold_pages, retrieved_pages)
        mrr_val = mrr(gold_pages, retrieved_pages)
        refused = refusal_detected(answer)

        # RAGAS metrics
        ragas_scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

        # Exclude pipeline errors from RAGAS scoring
        if not err and answer:
            sample = SingleTurnSample(
                user_input=q_text,
                response=answer,
                reference=gold_answer,
                retrieved_contexts=contexts if contexts else [""]
            )
            try:
                ragas_scores = asyncio.run(
                    score_sample_ragas(sample, is_unanswerable, metrics_dict)
                )
            except Exception as e:
                print(f"   -> Judge scoring error on {q_id}: {e}")

        row = {
            "id": q_id,
            "question": q_text,
            "type": q_type,
            "split": item.get("split", split),
            "ground_truth": gold_answer,
            "answer": answer,
            "latency_seconds": latency,
            "source_pages": json.dumps(gold_pages),
            "retrieved_pages": json.dumps(retrieved_pages),
            "hit_rate": hr if hr is not None else "N/A",
            "mrr": round(mrr_val, 4) if mrr_val is not None else "N/A",
            "refusal_detected": refused,
            "faithfulness": (
                round(ragas_scores["faithfulness"], 4)
                if ragas_scores["faithfulness"] is not None and not str(ragas_scores["faithfulness"]) == "nan"
                else "N/A"
            ),
            "answer_relevancy": (
                round(ragas_scores["answer_relevancy"], 4)
                if ragas_scores["answer_relevancy"] is not None and not str(ragas_scores["answer_relevancy"]) == "nan"
                else "N/A"
            ),
            "context_precision": (
                round(ragas_scores["context_precision"], 4)
                if ragas_scores["context_precision"] is not None and not str(ragas_scores["context_precision"]) == "nan"
                else "N/A"
            ),
            "context_recall": (
                round(ragas_scores["context_recall"], 4)
                if ragas_scores["context_recall"] is not None and not str(ragas_scores["context_recall"]) == "nan"
                else "N/A"
            ),
            "error": err or "",
        }
        results_rows.append(row)

        if idx < len(items):
            time.sleep(sleep_seconds)

    # 5. Output directories
    os.makedirs("results", exist_ok=True)
    per_question_file = f"results/{label}_per_question.csv"

    fieldnames = [
        "id", "question", "type", "split", "ground_truth", "answer",
        "latency_seconds", "source_pages", "retrieved_pages",
        "hit_rate", "mrr", "refusal_detected",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "error"
    ]
    with open(per_question_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nPer-question results written to: {per_question_file}")

    # 6. Aggregate Summary Metrics
    def safe_mean(key):
        vals = [r[key] for r in results_rows if r[key] != "N/A" and r[key] != ""]
        valid_floats = []
        for v in vals:
            try:
                valid_floats.append(float(v))
            except (ValueError, TypeError):
                pass
        return round(sum(valid_floats) / len(valid_floats), 4) if valid_floats else 0.0

    mean_latency = safe_mean("latency_seconds")
    mean_hit_rate = safe_mean("hit_rate")
    mean_mrr = safe_mean("mrr")
    mean_faith = safe_mean("faithfulness")
    mean_relevancy = safe_mean("answer_relevancy")
    mean_precision = safe_mean("context_precision")
    mean_recall = safe_mean("context_recall")

    refusal_stats = compute_refusal_rates(
        [{"source_pages": json.loads(r["source_pages"]), "type": r["type"], "answer": r["answer"]} for r in results_rows]
    )

    summary = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "n_questions": len(items),
        "n_errors": n_errors,
        "mean_latency": mean_latency,
        "mean_hit_rate": mean_hit_rate,
        "mean_mrr": mean_mrr,
        "correct_refusal_rate": refusal_stats["correct_refusal_rate"] or 0.0,
        "false_refusal_rate": refusal_stats["false_refusal_rate"] or 0.0,
        "mean_faithfulness": mean_faith,
        "mean_answer_relevancy": mean_relevancy,
        "mean_context_precision": mean_precision,
        "mean_context_recall": mean_recall,
        "k": k,
        "chunk_size": chunk_size,
        "chat_model": chat_model_name,
        "judge_model": f"{prov}:{judge_model_name}",
    }

    # Append to results/runs.csv
    runs_file = "results/runs.csv"
    runs_exists = os.path.exists(runs_file)
    with open(runs_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not runs_exists:
            writer.writeheader()
        writer.writerow(summary)
    print(f"Run summary appended to: {runs_file}")

    # 7. Print formatted summary table to console
    print("\n" + "=" * 75)
    print(f"EVALUATION SUMMARY: {label.upper()}")
    print("=" * 75)
    print(f"Questions Evaluated:   {summary['n_questions']} (Errors: {summary['n_errors']})")
    print(f"Mean Latency:          {summary['mean_latency']}s")
    print(f"Hit Rate @ {k}:         {summary['mean_hit_rate'] * 100:.1f}%")
    print(f"MRR:                   {summary['mean_mrr']:.4f}")
    print(f"Correct Refusal Rate:  {summary['correct_refusal_rate'] * 100:.1f}%")
    print(f"False Refusal Rate:    {summary['false_refusal_rate'] * 100:.1f}%")
    print(f"Faithfulness:          {summary['mean_faithfulness']:.4f}")
    print(f"Answer Relevancy:      {summary['mean_answer_relevancy']:.4f}")
    print(f"Context Precision:     {summary['mean_context_precision']:.4f}")
    print(f"Context Recall:        {summary['mean_context_recall']:.4f}")
    print(f"Models:                Chat={chat_model_name} | Judge={summary['judge_model']}")
    print("=" * 75 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline performance.")
    parser.add_argument("--label", type=str, required=True, help="Evaluation run label")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "heldout", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--persist-dir", type=str, default="chroma_db", help="Path to Chroma vector database")
    parser.add_argument("--dataset", type=str, default="eval_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--judge-provider", type=str, default=None, choices=["gemini", "mistral"], help="Judge LLM provider")
    args = parser.parse_args()

    run_evaluation(
        label=args.label,
        split=args.split,
        limit=args.limit,
        persist_dir=args.persist_dir,
        dataset_path=args.dataset,
        judge_provider=args.judge_provider,
    )

