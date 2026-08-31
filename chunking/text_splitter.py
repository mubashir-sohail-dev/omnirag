"""Text chunking module using RecursiveCharacterTextSplitter."""

import logging
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_SEPARATORS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split LangChain Document objects into overlapping chunks with metadata preservation.

    Args:
        documents: List of input Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.

    Returns:
        List of chunked Document objects with preserved metadata.

    Raises:
        ValueError: If *chunk_size* or *chunk_overlap* are invalid.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
    )

    logger.info("Chunking %d document(s)...", len(documents))
    chunked_docs = text_splitter.split_documents(documents)

    preview_path = "chunked_output_preview0.txt"
    with open(preview_path, "w", encoding="utf-8") as fh:
        for i, doc in enumerate(chunked_docs):
            src = doc.metadata.get("source", "unknown")
            fh.write(f"[CHUNK {i + 1} | Source: {src}]\n")
            fh.write(doc.page_content + "\n\n")

    logger.info(
        "Created %d chunks (preview saved to %s).", len(chunked_docs), preview_path
    )
    return chunked_docs


def chunk_data(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split a text file path into text string chunks.

    Args:
        file_path: Path to the input text file.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlapping characters between chunks.

    Returns:
        List of document text strings.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        raw_text = fh.read()

    doc = Document(page_content=raw_text, metadata={"source": file_path})
    chunked_docs = split_documents(
        [doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [d.page_content for d in chunked_docs]
