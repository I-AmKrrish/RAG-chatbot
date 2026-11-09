# Phase 01: Hybrid Retrieval & Re-ranking - Pattern Map

**Mapped:** 2026-11-10  
**Files analyzed:** 6 files (4 modified, 2 new)  
**Analogs found:** 5 / 6 (5 exact/role-match, 1 config with no codebase analog)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `rag_pipeline.py` | service | request-response | `rag_pipeline.py` (lines 81–146) | exact |
| `app.py` | component | request-response | `app.py` (lines 107–145, 207–254) | exact |
| `evaluate_rag.py` | utility / test | batch / request-response | `evaluate_rag.py` (lines 69–144, 348–366) | exact |
| `requirements.txt` | config | declarative | `requirements.txt` (lines 1–8, 114–119) | exact |
| `tests/test_retrieval.py` | test | batch / request-response | `tests/test_metrics.py` (lines 1–58) | role-match |
| `pytest.ini` | config | declarative | None (standard pytest configuration) | no-analog |

---

## Pattern Assignments

### `rag_pipeline.py` (service, request-response)

**Analog:** `rag_pipeline.py` (lines 1–147)

**Imports pattern** (lines 1–13):
Follow existing import structure, separating stdlib, path/env setup, and LangChain modules:
```python
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Safe environment loading without logging secrets
load_dotenv(Path(__file__).parent / ".env")

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
```
*Phase 1 Extension imports (RESEARCH.md verified):*
```python
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
# CRITICAL: Must import from langchain_classic.retrievers to prevent langchain_core.memory crash
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
```

**Retriever Construction & Dispatch Pattern** (lines 112–126):
Currently `build_pipeline` sets up retriever via branch:
```python
if search_type == "mmr":
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        }
    )
else:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
```
*Copy this structure to extract `build_retriever()` supporting `"hybrid"` mode alongside `"mmr"` and `"similarity"`:*
```python
def build_retriever(
    vectorstore: Chroma,
    retrieval_mode: str = "hybrid",
    k: int = 4,
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2",
):
    if retrieval_mode == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
    elif retrieval_mode == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    elif retrieval_mode == "hybrid":
        # 1. Build in-memory BM25 from Chroma chunks
        chroma_data = vectorstore.get()
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(
                chroma_data.get("documents", []), chroma_data.get("metadatas", [])
            )
        ]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = fetch_k

        # 2. Dense vector retriever fetching candidate pool
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": fetch_k},
        )

        # 3. Ensemble fusion (RRF)
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5],
        )

        # 4. Local cross-encoder re-ranking
        compressor = FlashrankRerank(model=reranker_model, top_n=k)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble,
        )
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")
```

**Pipeline Invocation & Page Metadata Preservation** (lines 24–28):
```python
# 1. Retrieval
docs = self.retriever.invoke(question)
contexts = [doc.page_content for doc in docs]
source_pages = [int(doc.metadata.get("page", 0)) + 1 for doc in docs]
context_str = "\n\n".join(contexts)
```

**Error Handling & Fallback Pattern** (lines 45–53):
```python
except Exception as e:
    latency = time.time() - start_time
    return {
        "answer": "",
        "contexts": [],
        "source_pages": [],
        "latency_seconds": round(latency, 3),
        "error": str(e),
    }
```
*Graceful Degradation Pattern:* In the hybrid compressor or pipeline wrapper, if `compressor.compress_documents()` raises an exception (or candidate pool is empty), fall back to returning `ensemble.invoke(question)[:k]` directly without raising an unhandled exception.

---

### `app.py` (component, request-response)

**Analog:** `app.py` (lines 107–145, 207–254)

**Sidebar Interface Pattern** (lines 107–120):
Add the retrieval mode selector to the sidebar directly above or beneath DB Directory:
```python
# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.title("🤖 Rag chatbot")
    if st.button("+ New Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    
    # DB Directory text input
    db_directory = st.text_input("Database Directory", value="chroma_db")

    # Retrieval Engine Toggle (Phase 1 D-06)
    retrieval_mode = st.radio(
        "Retrieval Engine",
        options=["Hybrid + Re-ranking", "Standard MMR"],
        index=0,
        help="Hybrid combines BM25 keyword search, Chroma vector search, and local FlashRank cross-encoder re-ranking."
    )
```

**Retriever Initialization & Session Execution** (lines 208–214, 235–254):
Replace direct hardcoded MMR retriever setup with `build_retriever()` call matching the selected sidebar mode:
```python
if st.session_state.db_ready:
    vectorstore = Chroma(persist_directory=db_directory, embedding_function=embedding_model)
    mode_key = "hybrid" if retrieval_mode == "Hybrid + Re-ranking" else "mmr"
    retriever = build_retriever(vectorstore=vectorstore, retrieval_mode=mode_key, k=4, fetch_k=10)
```

---

### `evaluate_rag.py` (utility / test, batch)

**Analog:** `evaluate_rag.py` (lines 69–144, 348–366)

**CLI Argument Parsing Pattern** (lines 348–366):
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline performance.")
    parser.add_argument("--label", type=str, required=True, help="Evaluation run label")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "heldout", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--persist-dir", type=str, default="chroma_db", help="Path to Chroma vector database")
    parser.add_argument("--dataset", type=str, default="eval_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--judge-provider", type=str, default=None, choices=["gemini", "mistral"], help="Judge LLM provider")
    # Phase 1 CLI extension:
    parser.add_argument("--search-type", type=str, default="mmr", choices=["mmr", "similarity", "hybrid"], help="Retrieval mode to evaluate")
    args = parser.parse_args()

    run_evaluation(
        label=args.label,
        split=args.split,
        limit=args.limit,
        persist_dir=args.persist_dir,
        dataset_path=args.dataset,
        judge_provider=args.judge_provider,
        search_type=args.search_type,
    )
```

**Runs CSV Summary Logging** (lines 298–326):
Summary dictionary must include `search_type` for comparative benchmarking in `results/runs.csv`:
```python
    summary = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "search_type": search_type,
        "n_questions": len(items),
        "n_errors": n_errors,
        "mean_latency": mean_latency,
        "mean_hit_rate": mean_hit_rate,
        "mean_mrr": mean_mrr,
        ...
    }
```

---

### `tests/test_retrieval.py` (test, batch)

**Analog:** `tests/test_metrics.py` (lines 1–58)

**Test File Imports Pattern** (lines 1–8):
```python
import pytest
from langchain_core.documents import Document
from rag_pipeline import build_retriever
```

**Unit Test Structure Pattern** (lines 10–25):
Clear arrange-act-assert pattern using mock/in-memory documents:
```python
def test_bm25_retriever_from_documents():
    docs = [
        Document(page_content="Gradient descent optimizes neural network parameters.", metadata={"page": 10}),
        Document(page_content="Convolutional layers capture spatial hierarchies.", metadata={"page": 25}),
        Document(page_content="Recurrent neural networks process sequential data.", metadata={"page": 40}),
    ]
    # Act: Test BM25 keyword matching for acronyms and terms
    # Assert chunk parity and citation metadata preservation
```

**Fallback / Graceful Degradation Test Pattern** (lines 52–58):
```python
def test_reranker_fallback_on_empty_or_failure():
    # Empty candidate pool should return empty list without throwing exception
    # Invalid model or scoring error should gracefully fall back to raw candidate list
```

---

### `requirements.txt` (config, declarative)

**Analog:** `requirements.txt` (lines 1–8, 114–119)

**Section Header and Dependency Pinning Style**:
```txt
# =========================
# Retrieval & Reranking
# =========================
rank-bm25==0.2.2
flashrank==0.2.10
huggingface-hub>=2.4.0
```

---

## Shared Patterns

### 1. Secret Protection & Environment Isolation
**Source:** `rag_pipeline.py` (lines 93–100), `evaluate_rag.py` (lines 82–95)  
**Apply to:** All scripts and pipeline factories. Never hardcode or print API keys. Validate presence and fail fast with clean human messages:
```python
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
```

### 2. 1-Indexed Source Page Citation Normalization
**Source:** `rag_pipeline.py` (line 27), `evaluate_rag.py` (line 181)  
**Apply to:** All components translating Chroma/PyPDF chunk metadata into user citations or benchmark metric inputs:
```python
source_pages = [int(doc.metadata.get("page", 0)) + 1 for doc in docs]
```

### 3. Graceful Degradation / Error Containment
**Source:** `rag_pipeline.py` (lines 45–53), `evaluate_rag.py` (lines 185–188)  
**Apply to:** All retrieval and compression operations. If re-ranking or retrieval fails, log error diagnostic and return degraded results without crashing the application or aborting batch benchmark runs.

### 4. LangChain Namespace Decoupling
**Source:** Verified runtime finding in `01-RESEARCH.md` (lines 44, 280–285)  
**Apply to:** All imports of `EnsembleRetriever` and `ContextualCompressionRetriever`.  
**Rule:** Always import from `langchain_classic.retrievers` instead of `langchain.retrievers` to prevent the `langchain_core.memory` `ModuleNotFoundError`.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `pytest.ini` | config | declarative | Standard pytest root configuration file. Does not currently exist in the repository; follows canonical pytest syntax (`pythonpath = .`, `testpaths = tests`). |

---

## Metadata

**Analog search scope:** Project root, `tests/`, `.planning/codebase/`  
**Files scanned:** 14  
**Pattern extraction date:** 2026-11-10
