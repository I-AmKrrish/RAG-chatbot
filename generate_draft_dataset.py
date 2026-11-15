import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Safe environment loading without logging secrets
load_dotenv(Path(__file__).parent / ".env")

from langchain_community.document_loaders import PyPDFLoader
from judge import get_judge_llm
from langchain_core.prompts import ChatPromptTemplate


def generate_drafts(pdf_path: str, output_path: str, num_samples: int = 10):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    total_pages = len(pages)
    print(f"Loaded {total_pages} pages.")

    # Use Judge LLM (Gemini or fallback Mistral)
    judge_llm = get_judge_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert dataset creator for RAG evaluation.
Given the text from specific pages of a textbook, generate candidate evaluation items in valid JSON format.
Each item must have:
- "question": a specific question based on the content
- "ground_truth": a concise factual answer
- "source_pages": list of 1-indexed page numbers where the answer is found
- "type": "simple" (answer from 1 page) or "multi_chunk" (concept spans multiple pages)
- "split": "dev" (80%) or "heldout" (20%)
- "status": "draft"

Also generate at least 2 "unanswerable" items:
- "question": a realistic question about machine learning that is NOT mentioned in these pages
- "ground_truth": "I could not find the answer in the given documents."
- "source_pages": []
- "type": "unanswerable"
- "split": "dev" or "heldout"
- "status": "draft"

Respond with ONLY a raw JSON array of objects. Do not include markdown codeblocks or extra text.
"""),
        ("human", "Pages content:\n\n{content}")
    ])

    # Sample a few diverse pages
    sample_indices = [10, 15, 20, 25, 30, 45, 60][:min(total_pages, 7)]
    sampled_content = ""
    for idx in sample_indices:
        p_num = idx + 1
        text_snip = pages[idx].page_content[:600].replace("\n", " ")
        sampled_content += f"[Page {p_num}]: {text_snip}\n\n"

    print("Generating candidate Q&A drafts using Judge LLM...")
    response = judge_llm.invoke(prompt.format(content=sampled_content))
    raw_content = response.content.strip()

    # Clean markdown fences if any
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
    raw_content = raw_content.strip()

    try:
        items = json.loads(raw_content)
    except Exception as e:
        print(f"Error parsing LLM response as JSON: {e}")
        items = []

    # Assign IDs and enforce draft status
    for i, it in enumerate(items, start=1):
        it["id"] = f"draft_{i:03d}"
        it["status"] = "draft"

    print(f"Generated {len(items)} draft items.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    print(f"Drafts written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft Q&A dataset items using the Judge LLM.")
    parser.add_argument("--pdf", type=str, default="document_loaders/deeplearning.pdf", help="Path to reference PDF")
    parser.add_argument("--output", type=str, default="eval_dataset_draft.json", help="Path to output JSON")
    args = parser.parse_args()

    generate_drafts(args.pdf, args.output)

