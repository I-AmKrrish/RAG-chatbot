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


class Pipeline:
    def __init__(self, retriever, llm, prompt_template):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template

    def answer(self, question: str) -> dict:
        start_time = time.time()
        try:
            # 1. Retrieval
            docs = self.retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
            source_pages = [int(doc.metadata.get("page", 0)) + 1 for doc in docs]
            context_str = "\n\n".join(contexts)

            # 2. Generation with Retry / Exponential Backoff
            final_prompt = self.prompt_template.invoke({
                "context": context_str,
                "question": question
            })

            answer_text = self._call_llm_with_retry(final_prompt)
            latency = time.time() - start_time
            return {
                "answer": answer_text,
                "contexts": contexts,
                "source_pages": source_pages,
                "latency_seconds": round(latency, 3),
            }

        except Exception as e:
            latency = time.time() - start_time
            return {
                "answer": "",
                "contexts": [],
                "source_pages": [],
                "latency_seconds": round(latency, 3),
                "error": str(e),
            }

    def _call_llm_with_retry(self, prompt, max_retries: int = 5, initial_wait: float = 2.0) -> str:
        wait_time = initial_wait
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return getattr(response, "content", str(response))
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str

                # Check for Retry-After if available
                retry_after = getattr(e, "retry_after", None)
                if hasattr(e, "response") and getattr(e.response, "headers", None):
                    retry_after = e.response.headers.get("Retry-After")

                if is_rate_limit and attempt < max_retries - 1:
                    sleep_dur = float(retry_after) if retry_after else wait_time
                    time.sleep(sleep_dur)
                    wait_time *= 2.0
                elif attempt < max_retries - 1 and not is_rate_limit:
                    time.sleep(wait_time)
                    wait_time *= 1.5
                else:
                    raise e


def build_pipeline(
    persist_dir: str = "chroma_db",
    k: int = 4,
    search_type: str = "mmr",
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
    model: str = None
) -> Pipeline:
    # Resolve Model
    if model is None:
        model = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment or .env file.")

    # Always use MistralAIEmbeddings to match chroma_db
    embedding_model = MistralAIEmbeddings(mistral_api_key=mistral_api_key)

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Chroma DB directory '{persist_dir}' does not exist.")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

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

    llm = ChatGroq(
        model=model,
        groq_api_key=groq_api_key,
        temperature=0
    )

    # Prompt verbatim from main.py
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant
        use ONLY the provided context to answer the question.
        If the answer is not in the context, say: "I could not find the answer in the given documents" """
        ),
        ("human",
        """Context: {context}
        Question: {question}
        """
        ),
    ])

    return Pipeline(retriever=retriever, llm=llm, prompt_template=prompt_template)

