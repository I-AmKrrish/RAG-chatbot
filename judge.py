import os
from pathlib import Path
from dotenv import load_dotenv

# Safe environment loading without logging secrets
load_dotenv(Path(__file__).parent / ".env")

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def get_judge_provider_and_model(provider: str = None, model: str = None):
    prov = (provider or os.getenv("JUDGE_PROVIDER", "gemini")).strip().lower()
    if prov == "gemini":
        mod = model or os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
    elif prov == "mistral":
        mod = model or os.getenv("JUDGE_MODEL", "open-mistral-7b")
    else:
        raise ValueError(
            f"Unsupported JUDGE_PROVIDER: '{prov}'. The judge must be a different provider "
            "from the Groq chatbot. Supported providers: 'gemini', 'mistral'."
        )
    return prov, mod


def get_judge_llm(provider: str = None, model: str = None):
    prov, mod = get_judge_provider_and_model(provider, model)

    if prov == "gemini":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing from environment or .env file. "
                "Required when JUDGE_PROVIDER=gemini."
            )
        return ChatGoogleGenerativeAI(
            model=mod,
            google_api_key=google_api_key,
            temperature=0
        )
    elif prov == "mistral":
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError(
                "MISTRAL_API_KEY is missing from environment or .env file. "
                "Required when JUDGE_PROVIDER=mistral."
            )
        return ChatMistralAI(
            model=mod,
            mistral_api_key=mistral_api_key,
            temperature=0
        )


def get_judge_embeddings():
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY is missing from environment or .env file.")
    return MistralAIEmbeddings(mistral_api_key=mistral_api_key)


def get_ragas_judge(provider: str = None, model: str = None):
    llm = get_judge_llm(provider=provider, model=model)
    embeddings = get_judge_embeddings()
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

