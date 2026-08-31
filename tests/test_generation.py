"""Unit tests for configuration settings, prompts, and LLM provider factory."""

import pytest
from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PROVIDER,
    EMBEDDING_MODEL,
)
from llm.prompts import RAG_PROMPT, SYSTEM_PROMPT
from llm.providers import get_llm


class TestSettingsAndPrompts:
    """Verify configuration settings and prompt invariants."""

    def test_system_prompt_has_context_placeholder(self):
        """System prompt must include context placeholder."""
        assert "{context}" in SYSTEM_PROMPT

    def test_system_prompt_has_refusal_pattern(self):
        """System prompt must mandate refusal if context is insufficient."""
        assert "I cannot answer" in SYSTEM_PROMPT

    def test_embedding_model_is_string(self):
        """Embedding model name must be non-empty string."""
        assert isinstance(EMBEDDING_MODEL, str)
        assert len(EMBEDDING_MODEL) > 0

    def test_chunk_params_valid(self):
        """Default chunk size must exceed overlap."""
        assert DEFAULT_CHUNK_SIZE > DEFAULT_CHUNK_OVERLAP
        assert DEFAULT_CHUNK_SIZE > 0
        assert DEFAULT_CHUNK_OVERLAP >= 0

    def test_rag_prompt_exists(self):
        """RAG prompt template must be defined."""
        assert RAG_PROMPT is not None

    def test_default_provider_valid(self):
        """Default provider must be google, groq, or ollama."""
        assert DEFAULT_PROVIDER in {"google", "groq", "ollama"}


class TestProviderFactory:
    """Test LLM provider factory `get_llm`."""

    def test_unsupported_provider_raises(self):
        """ValueError when an invalid LLM provider is requested."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm("invalid_provider")
