import os
import sys
import json
import csv
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Safe environment loading without logging secrets
load_dotenv(Path(__file__).parent / ".env")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

from evaluate_rag import run_evaluation


def clean_text(text: str) -> str:
    if not text:
        return ""
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


def build_eval_database(pdf_path: str, chunk_size: int, eval_dir: str):
    """
    Builds an isolated evaluation Chroma database at eval_dbs/cs<size>/.
    Never touches ./chroma_db.
    """
    if os.path.exists(eval_dir) and len(os.listdir(eval_dir)) > 0:
        print(f"Directory '{eval_dir}' already exists. Skipping database build.")
        return

    print(f"Building isolated Chroma DB for chunk_size={chunk_size} at '{eval_dir}'...")
    os.makedirs(eval_dir, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    cleaned_docs = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content.strip():
            doc.page_content = cleaned_content
            cleaned_docs.append(doc)

    overlap = int(chunk_size * 0.1)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(cleaned_docs)
    print(f"Generated {len(chunks)} chunks for chunk_size={chunk_size}.")

    mistral_key = os.getenv("MISTRAL_API_KEY")
    embedding_model = MistralAIEmbeddings(mistral_api_key=mistral_key)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=eval_dir
    )
    print(f"Database build complete at '{eval_dir}'.")


def get_preset_configs(preset: str):
    if preset == "chunking":
        return [
            {"chunk_size": 500, "k": 4, "search_type": "mmr", "name": "chunking_cs500"},
            {"chunk_size": 1000, "k": 4, "search_type": "mmr", "name": "chunking_cs1000"},
            {"chunk_size": 1500, "k": 4, "search_type": "mmr", "name": "chunking_cs1500"},
        ]
    elif preset == "k":
        return [
            {"chunk_size": 1000, "k": 3, "search_type": "mmr", "name": "k_k3"},
            {"chunk_size": 1000, "k": 4, "search_type": "mmr", "name": "k_k4"},
            {"chunk_size": 1000, "k": 6, "search_type": "mmr", "name": "k_k6"},
        ]
    elif preset == "search":
        return [
            {"chunk_size": 1000, "k": 4, "search_type": "mmr", "name": "search_mmr"},
            {"chunk_size": 1000, "k": 4, "search_type": "similarity", "name": "search_similarity"},
        ]
    else:
        raise ValueError(f"Unknown preset: '{preset}'. Valid presets: 'chunking', 'k', 'search'.")


def generate_comparison_report(preset: str, completed_labels: list):
    runs_file = "results/runs.csv"
    if not os.path.exists(runs_file):
        print(f"Cannot generate comparison report: '{runs_file}' does not exist.")
        return

    with open(runs_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_runs = list(reader)

    # Filter runs for this preset
    preset_runs = [r for r in all_runs if r["label"] in completed_labels]
    if not preset_runs:
        print(f"No completed runs found matching preset '{preset}'.")
        return

    # Metrics where higher is better
    higher_better = [
        "mean_hit_rate", "mean_mrr", "correct_refusal_rate",
        "mean_faithfulness", "mean_answer_relevancy",
        "mean_context_precision", "mean_context_recall"
    ]
    # Metrics where lower is better
    lower_better = ["false_refusal_rate", "mean_latency"]

    all_metric_keys = higher_better + lower_better

    # Find best values per metric
    best_vals = {}
    for k in all_metric_keys:
        vals = [float(r[k]) for r in preset_runs if r[k] not in ("", "N/A")]
        if not vals:
            continue
        best_vals[k] = max(vals) if k in higher_better else min(vals)

    # Build Markdown table
    headers = [
        "Label", "Chunk Size", "k", "Hit Rate", "MRR",
        "Correct Refusal", "False Refusal", "Faithfulness",
        "Relevancy", "Context Precision", "Context Recall", "Latency (s)"
    ]

    lines = []
    lines.append(f"# Experiment Comparison Report: Preset `{preset.upper()}`\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in preset_runs:
        def fmt(key, is_pct=False):
            raw = r.get(key, "N/A")
            try:
                val = float(raw)
            except (ValueError, TypeError):
                return raw
            is_best = (key in best_vals and val == best_vals[key])
            text = f"{val * 100:.1f}%" if is_pct else f"{val:.4f}"
            return f"**{text}**" if is_best else text

        row_str = (
            f"| `{r['label']}` "
            f"| {r.get('chunk_size', '1000')} "
            f"| {r.get('k', '4')} "
            f"| {fmt('mean_hit_rate', True)} "
            f"| {fmt('mean_mrr')} "
            f"| {fmt('correct_refusal_rate', True)} "
            f"| {fmt('false_refusal_rate', True)} "
            f"| {fmt('mean_faithfulness')} "
            f"| {fmt('mean_answer_relevancy')} "
            f"| {fmt('mean_context_precision')} "
            f"| {fmt('mean_context_recall')} "
            f"| {fmt('mean_latency')} |"
        )
        lines.append(row_str)

    # Find best overall config (by sum of normalized rank or highest hit rate + faithfulness)
    best_overall = max(
        preset_runs,
        key=lambda r: float(r.get("mean_hit_rate", 0)) + float(r.get("mean_faithfulness", 0))
    )
    best_overall_label = best_overall["label"]

    # Identify metrics that got worse in the best-overall config
    tradeoffs = []
    for k in all_metric_keys:
        if k in best_vals:
            val = float(best_overall.get(k, 0))
            if val != best_vals[k]:
                metric_name = k.replace("mean_", "").replace("_", " ").title()
                tradeoffs.append(f"- **{metric_name}**: {val} vs best {best_vals[k]}")

    lines.append("\n## Analysis & Trade-offs\n")
    lines.append(f"- **Top-Performing Overall Configuration**: `{best_overall_label}`")
    if tradeoffs:
        lines.append("- **Metrics that sacrificed performance in the top configuration**:")
        lines.extend(tradeoffs)
    else:
        lines.append("- The top-performing configuration achieved the best score across all evaluated metrics!")

    report_path = f"results/comparison_{preset}.md"
    os.makedirs("results", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nComparison report written to: {report_path}")
    print("\n" + "\n".join(lines) + "\n")


def run_experiments(
    preset: str,
    pdf_path: str = "document_loaders/deeplearning.pdf",
    limit: int = 5,
    yes: bool = False,
    resume: bool = False,
    judge_provider: str = None
):
    configs = get_preset_configs(preset)

    # Read dataset to count dev questions
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dev_items = [it for it in dataset if it.get("status") == "reviewed" and it.get("split") == "dev"]
    num_questions = min(len(dev_items), limit) if limit else len(dev_items)

    # Check for already completed runs if resuming
    existing_labels = set()
    if os.path.exists("results/runs.csv"):
        with open("results/runs.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_labels.add(r["label"])

    active_configs = []
    for c in configs:
        if resume and c["name"] in existing_labels:
            print(f"Skipping already recorded run: {c['name']}")
        else:
            active_configs.append(c)

    if not active_configs:
        print("All experiment runs in this preset are already completed.")
        generate_comparison_report(preset, [c["name"] for c in configs])
        return

    # Call Estimates
    # Check which eval_dbs already exist
    new_db_chunk_sizes = set()
    for c in active_configs:
        cs = c["chunk_size"]
        db_path = f"eval_dbs/cs{cs}"
        if not os.path.exists(db_path) or len(os.listdir(db_path)) == 0:
            new_db_chunk_sizes.add(cs)

    # Approximate ~700 chunks for cs=1000
    est_embed_calls = sum(int(700 * (1000 / cs)) for cs in new_db_chunk_sizes)
    est_groq_calls = num_questions * len(active_configs)
    est_judge_calls = num_questions * len(active_configs) * 4

    print("\n" + "=" * 65)
    print("EXPERIMENT CALL & COST ESTIMATES")
    print("=" * 65)
    print(f"  Preset:                   {preset}")
    print(f"  Split:                    dev (strictly)")
    print(f"  Questions per run:        {num_questions}")
    print(f"  Runs to execute:          {len(active_configs)}")
    print(f"  Estimated Embeddings:     ~{est_embed_calls} (new Chroma builds: {list(new_db_chunk_sizes)})")
    print(f"  Estimated Groq LLM Calls: ~{est_groq_calls}")
    print(f"  Estimated Judge Calls:    ~{est_judge_calls}")
    print("=" * 65)

    if not yes:
        confirm = input("Proceed with experiment execution? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted by user.")
            sys.exit(0)

    # Execute runs
    completed_labels = []
    for c in configs:
        label = c["name"]
        cs = c["chunk_size"]
        k = c["k"]
        st = c["search_type"]

        if resume and label in existing_labels:
            completed_labels.append(label)
            continue

        # Build isolated DB if needed
        eval_db_path = f"eval_dbs/cs{cs}"
        build_eval_database(pdf_path, cs, eval_db_path)

        print(f"\n>>> Running Experiment: {label} (cs={cs}, k={k}, search={st}) <<<")
        run_evaluation(
            label=label,
            split="dev",
            limit=limit,
            persist_dir=eval_db_path,
            dataset_path="eval_dataset.json",
            judge_provider=judge_provider,
            k=k,
            search_type=st,
            chunk_size=cs,
        )
        completed_labels.append(label)

    # Generate final comparison report
    generate_comparison_report(preset, completed_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG hyperparameter experiments.")
    parser.add_argument("--preset", type=str, required=True, choices=["chunking", "k", "search"])
    parser.add_argument("--pdf", type=str, default="document_loaders/deeplearning.pdf", help="Path to PDF")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of questions to evaluate")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--resume", action="store_true", help="Resume and skip previously recorded runs")
    parser.add_argument("--judge-provider", type=str, default=None, choices=["gemini", "mistral"])
    args = parser.parse_args()

    run_experiments(
        preset=args.preset,
        pdf_path=args.pdf,
        limit=args.limit,
        yes=args.yes,
        resume=args.resume,
        judge_provider=args.judge_provider
    )

