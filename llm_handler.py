"""
llm_handler.py – Modular LLM and Embeddings factory.

Supports three interchangeable backends with zero code changes in the rest of
the application:
  • OpenAI   – GPT-4o, GPT-4o-mini, GPT-3.5-turbo, etc.
  • Anthropic – claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5, etc.
  • Ollama   – Any locally served model (llama3, mistral, phi3, etc.)

Embedding backends:
  • OpenAI             – text-embedding-3-small / text-embedding-3-large
  • HuggingFace (local) – BAAI/bge-small-en-v1.5 (default, free, fast)

Switching providers: change LLM_PROVIDER / EMBEDDING_PROVIDER in your .env.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Union

from langchain.schema.language_model import BaseLanguageModel
from langchain.embeddings.base import Embeddings

from config import cfg

logger = logging.getLogger(__name__)


# ── Type aliases ──────────────────────────────────────────────────────────────

LLMType = BaseLanguageModel
EmbeddingType = Embeddings


# ── LLM Factory ───────────────────────────────────────────────────────────────

def get_llm(
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMType:
    """
    Instantiate and return a LangChain-compatible LLM.

    Parameters
    ----------
    provider : str, optional
        Override cfg.llm_provider. One of "openai", "anthropic", "ollama".
    temperature : float, optional
        Override cfg.llm_temperature.
    max_tokens : int, optional
        Override cfg.llm_max_tokens.

    Returns
    -------
    BaseChatModel
        A LangChain chat model ready for invoke() / stream() calls.

    Raises
    ------
    ValueError
        If the requested provider is unknown.
    ImportError
        If the required langchain integration package is missing.
    """
    provider = (provider or cfg.llm_provider).lower()
    temp = temperature if temperature is not None else cfg.llm_temperature
    max_tok = max_tokens or cfg.llm_max_tokens

    logger.info("Initialising LLM: provider=%s", provider)

    if provider == "openai":
        return _make_openai_llm(temp, max_tok)
    elif provider == "anthropic":
        return _make_anthropic_llm(temp, max_tok)
    elif provider == "ollama":
        return _make_ollama_llm(temp, max_tok)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Choose from: openai, anthropic, ollama"
        )


def _make_openai_llm(temperature: float, max_tokens: int) -> LLMType:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("Run: pip install langchain-openai") from exc

    if not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    return ChatOpenAI(
        model=cfg.openai_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=cfg.openai_api_key,
    )


def _make_anthropic_llm(temperature: float, max_tokens: int) -> LLMType:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError("Run: pip install langchain-anthropic") from exc

    if not cfg.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    return ChatAnthropic(
        model=cfg.anthropic_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=cfg.anthropic_api_key,
    )


def _make_ollama_llm(temperature: float, max_tokens: int) -> LLMType:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError("Run: pip install langchain-ollama") from exc

    return ChatOllama(
        model=cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )


# ── Embeddings Factory ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings(provider: str | None = None) -> EmbeddingType:
    """
    Return a (cached) LangChain Embeddings object.

    Uses lru_cache so the (potentially heavy) model is loaded only once per
    Python process.

    Parameters
    ----------
    provider : str, optional
        Override cfg.embedding_provider. One of "openai", "huggingface".
    """
    provider = (provider or cfg.embedding_provider).lower()
    logger.info("Initialising embeddings: provider=%s", provider)

    if provider == "openai":
        return _make_openai_embeddings()
    elif provider == "huggingface":
        return _make_hf_embeddings()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. "
            "Choose from: openai, huggingface"
        )


def _make_openai_embeddings() -> EmbeddingType:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError("Run: pip install langchain-openai") from exc

    if not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=cfg.openai_api_key,
    )


def _make_hf_embeddings() -> EmbeddingType:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError("Run: pip install sentence-transformers") from exc

    return HuggingFaceEmbeddings(
        model_name=cfg.hf_embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Streaming helper ──────────────────────────────────────────────────────────

def stream_llm_response(llm: LLMType, messages: list) -> str:
    """
    Stream tokens from the LLM and return the full concatenated response.

    This is a thin wrapper to allow callers to iterate over chunks for a
    live-update UI without directly coupling to LangChain internals.
    """
    full_response = ""
    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", str(chunk))
        full_response += content
    return full_response
