# llm_factory.py
"""
Factory functions to create LangChain-compatible LLM/chat models
for different providers, used by app.py and nl2sql_runner.py.

Supported provider names (UI labels):
  - "OpenAI GPT-4o-mini"
  - "Gemini 2.5 Flash"
  - "Mistral Large (API)"
  - "Claude 3.5"
  - "Ollama Llama3"
  - "Local T5 (Spider)"
"""

import os

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

def _require_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(
            f"{var_name} is not set. Please export it (or put it in .env) before running the app."
        )
    return value


# -------------------------------------------------------------------
# Individual providers
# -------------------------------------------------------------------
def _make_openai() -> ChatOpenAI:
    _require_env("OPENAI_API_KEY")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=60,
        max_retries=2,
        # api key read from OPENAI_API_KEY
    )

def _make_gemini():
    """
    Google Gemini via langchain-google-genai.
    Uses GOOGLE_API_KEY from env.
    This variant is compatible with:
      - langchain-google-genai==1.0.4
      - google-generativeai==0.5.4
    """
    # pip install -U langchain-google-genai==1.0.4 google-generativeai==0.5.4
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]

    _require_env("GOOGLE_API_KEY")

    # For this SDK version, the correct model IDs are "gemini-1.5-flash-001", "gemini-1.5-pro-001", etc.
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_output_tokens=2048,
        max_retries=2,
        # api_key is read from GOOGLE_API_KEY env var
    )



def _make_mistral_api():
    # pip install langchain-mistralai mistralai
    from langchain_mistralai import ChatMistralAI  # type: ignore[import]

    _require_env("MISTRAL_API_KEY")
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.0,
        max_retries=2,
    )


def _make_claude():
    # pip install langchain-anthropic anthropic
    from langchain_anthropic import ChatAnthropic  # type: ignore[import]

    _require_env("ANTHROPIC_API_KEY")
    return ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0.0,
        max_tokens=None,
    )


def _make_ollama(model_name: str = "llama3"):
    # pip install langchain-ollama
    from langchain_ollama import ChatOllama  # type: ignore[import]

    return ChatOllama(
        model=model_name,
        temperature=0.0,
    )


def _make_local_t5():
    """
    Local T5 text-to-SQL (Spider fine-tuned).
    - If HF_LOCAL_MODEL_DIR is set, loads from that path.
    - Else downloads from Hugging Face.
    Requires transformers + a backend (usually torch).
    """
    try:
        from transformers import pipeline  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "transformers is not installed. Install it if you want to use 'Local T5 (Spider)'."
        )

    model_ref = os.getenv("HF_LOCAL_MODEL_DIR", "").strip() or "tscholak/text-to-sql-t5-large"
    gen = pipeline(
        "text2text-generation",
        model=model_ref,
        max_new_tokens=256,
        temperature=0.0,
    )
    return HuggingFacePipeline(pipeline=gen)


# -------------------------------------------------------------------
# Public dispatch
# -------------------------------------------------------------------
def get_llm(provider_name: str):
    name = (provider_name or "").lower()

    if "openai" in name:
        return _make_openai()

    if "gemini" in name:
        return _make_gemini()

    if "mistral" in name and "api" in name:
        return _make_mistral_api()

    if "claude" in name:
        return _make_claude()

    if "ollama" in name:
        return _make_ollama("llama3")

    if "local t5" in name or "spider" in name:
        return _make_local_t5()

    # Fallback
    return _make_openai()
