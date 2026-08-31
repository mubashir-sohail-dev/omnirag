"""Unit tests for ChromaDB vector store management."""

import pytest
from vectorstore.chroma_store import create_vector_store


class TestCreateVectorStoreValidation:
    """Input validation tests for ``create_vector_store``."""

    def test_empty_chunks_raises(self):
        """ValueError when chunks or documents list is empty."""
        with pytest.raises(ValueError, match="empty"):
            create_vector_store([])
