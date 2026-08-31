"""Unit tests for unified document ingestion loader (native PDF, OCR fallback, TXT, directories)."""

import pytest
from ingestion.document_loader import load_documents, load_pdf_file, load_txt_file
from utils.helpers import discover_supported_files


class TestIngestionDiscovery:
    """Test file discovery and path validation."""

    def test_single_file_discovery(self, sample_text_file):
        """Discovering a single text file returns a list with that file."""
        files = discover_supported_files(sample_text_file)
        assert len(files) == 1

    def test_directory_discovery(self, sample_directory):
        """Discovering a directory returns all supported document files."""
        files = discover_supported_files(sample_directory)
        assert len(files) == 2

    def test_unsupported_file_raises(self, tmp_path):
        """ValueError when single file has unsupported extension."""
        unsupported = tmp_path / "image.png"
        unsupported.write_bytes(b"\x89PNG")
        with pytest.raises(ValueError, match="Unsupported file format"):
            discover_supported_files(str(unsupported))

    def test_empty_directory_raises(self, tmp_path):
        """ValueError when directory contains no supported files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No supported document files"):
            discover_supported_files(str(empty_dir))


class TestTextLoader:
    """Test plain text file loader with metadata tracking."""

    def test_load_text_file_content_and_metadata(self, sample_text_file):
        """`load_txt_file` returns a Document object with content and metadata."""
        docs = load_txt_file(sample_text_file)
        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content.startswith("[PAGE 1]")
        assert doc.metadata["file_type"] == "txt"
        assert doc.metadata["page"] == 1
        assert doc.metadata["total_pages"] == 1
        assert doc.metadata["ocr_used"] is False


class TestUnifiedLoader:
    """Test unified document ingestion loader."""

    def test_load_documents_from_file(self, sample_text_file):
        """`load_documents` ingests a single text file."""
        docs = load_documents(sample_text_file)
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "txt"

    def test_load_documents_from_directory(self, sample_directory):
        """`load_documents` ingests all files in a directory."""
        docs = load_documents(sample_directory)
        assert len(docs) == 2


class TestPdfLoaderValidation:
    """Input validation for PDF loader."""

    def test_missing_pdf_raises(self):
        """FileNotFoundError when PDF path does not exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_pdf_file("nonexistent_file.pdf")
