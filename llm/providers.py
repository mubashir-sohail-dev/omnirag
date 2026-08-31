"""Unified LLM provider factory module supporting Google, Groq, and Ollama."""

import logging
from typing import Optional
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from config.settings import DEFAULT_MODEL_MAP, DEFAULT_PROVIDER

logger = logging.getLogger(__name__)


def get_llm(
    provider: str = DEFAULT_PROVIDER,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
):
    """Instantiate and return a configured LLM client.

    Args:
        provider: Provider identifier ('google', 'groq', 'ollama').
        model_name: Optional model override. Defaults to value in settings.py.
        temperature: Model sampling temperature (default: 0.0).

    Returns:
        Configured LangChain LLM / ChatModel instance.

    Raises:
        ValueError: If *provider* is not supported.
    """
    provider_clean = provider.strip().lower()
    selected_model = model_name or DEFAULT_MODEL_MAP.get(provider_clean)

    if not selected_model:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Supported providers: "
            f"{', '.join(sorted(DEFAULT_MODEL_MAP.keys()))}"
        )

    logger.info(
        "Initializing LLM client [Provider: %s | Model: %s]",
        provider_clean,
        selected_model,
    )

    if provider_clean == "google":
        return ChatGoogleGenerativeAI(
            model=selected_model, temperature=temperature
        )
    elif provider_clean == "groq":
        return ChatGroq(model=selected_model, temperature=temperature)
    elif provider_clean == "ollama":
        return Ollama(model=selected_model, temperature=temperature)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. Supported providers: "
            f"{', '.join(sorted(DEFAULT_MODEL_MAP.keys()))}"
        )
