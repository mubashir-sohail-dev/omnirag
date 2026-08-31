"""Pytest configuration and shared fixtures for custom_rag."""

import os
import sys
import pytest

# Ensure custom_rag root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_text_file(tmp_path):
    """Fixture providing a temporary text file with sample document content."""
    text_content = (
        "[PAGE 1]\n"
        "Data Communication Principles and Network Protocols.\n"
        "Data transmission occurs over physical media using signals.\n\n"
        "[PAGE 2]\n"
        "Routing algorithms determine the optimal path for data packets.\n"
        "TCP ensures reliable transmission through sequence numbers and ACKs.\n"
    )
    file_path = tmp_path / "sample_document.txt"
    file_path.write_text(text_content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_directory(tmp_path):
    """Fixture providing a directory containing multiple sample files."""
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text("First sample document content.", encoding="utf-8")
    (doc_dir / "doc2.txt").write_text("Second sample document content.", encoding="utf-8")
    return str(doc_dir)


@pytest.fixture
def sample_chunks():
    """Fixture providing a pre-chunked list of text strings."""
    return [
        "Data Communication Principles and Network Protocols.",
        "Data transmission occurs over physical media using signals.",
        "Routing algorithms determine the optimal path for data packets.",
        "TCP ensures reliable transmission through sequence numbers and ACKs.",
    ]
