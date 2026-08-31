"""Unit tests for text chunking module."""

import os
import pytest
from chunking.text_splitter import chunk_data, split_documents
from ingestion.document_loader import load_documents


class TestChunkDataValidation:
    """Input validation tests for chunking functions."""

    def test_file_not_found_raises(self):
        """FileNotFoundError when input file path does not exist."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            chunk_data("non_existent_file.txt")

    def test_zero_chunk_size_raises(self, sample_text_file):
        """ValueError when chunk_size <= 0."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_data(sample_text_file, chunk_size=0)

    def test_negative_chunk_size_raises(self, sample_text_file):
        """ValueError when chunk_size < 0."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_data(sample_text_file, chunk_size=-100)

    def test_negative_overlap_raises(self, sample_text_file):
        """ValueError when chunk_overlap < 0."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunk_data(sample_text_file, chunk_overlap=-10)

    def test_overlap_exceeds_size_raises(self, sample_text_file):
        """ValueError when chunk_overlap >= chunk_size."""
        with pytest.raises(ValueError, match="must be less than"):
            chunk_data(sample_text_file, chunk_size=100, chunk_overlap=100)


class TestChunkDataBehaviour:
    """Behavioral tests for document chunking."""

    def test_produces_non_empty_chunks(self, sample_text_file):
        """Chunk list returned must be non-empty."""
        chunks = chunk_data(sample_text_file, chunk_size=100, chunk_overlap=20)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_split_documents_preserves_metadata(self, sample_text_file):
        """`split_documents` preserves Document metadata."""
        docs = load_documents(sample_text_file)
        chunked = split_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunked) > 0
        assert chunked[0].metadata["source"] == os.path.abspath(sample_text_file)
        assert chunked[0].metadata["file_type"] == "txt"

    def test_preview_file_created(self, sample_text_file):
        """A preview file chunked_output_preview0.txt is written."""
        preview_path = "chunked_output_preview0.txt"
        if os.path.exists(preview_path):
            os.remove(preview_path)

        chunk_data(sample_text_file, chunk_size=100, chunk_overlap=20)
        assert os.path.exists(preview_path)
