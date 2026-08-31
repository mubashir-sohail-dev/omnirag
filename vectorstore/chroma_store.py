"""ChromaDB vector store management module.

Combines creation, loading, persistence, and validation into a unified interface.
"""

import logging
from typing import Union
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import DEFAULT_VECTOR_DB_PATH
from embeddings.dense_embeddings import get_dense_embeddings

logger = logging.getLogger(__name__)


def create_vector_store(
    documents_or_chunks: Union[list[Document], list[str]],
    persist_directory: str = DEFAULT_VECTOR_DB_PATH,
) -> Chroma:
    """Create and persist a ChromaDB vector store from Document objects or text chunks.

    Args:
        documents_or_chunks: List of LangChain Document objects or raw text strings.
        persist_directory: Path to store persistent Chroma database.

    Returns:
        Created :class:`Chroma` database instance.

    Raises:
        ValueError: If *documents_or_chunks* is empty.
    """
    if not documents_or_chunks:
        raise ValueError("Cannot create vector store from an empty list.")

    logger.info("Creating Chroma vector store in %s ...", persist_directory)
    embeddings = get_dense_embeddings()

    if isinstance(documents_or_chunks[0], Document):
        logger.info("Indexing %d Document objects...", len(documents_or_chunks))
        vector_store = Chroma.from_documents(
            documents_or_chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
    else:
        logger.info("Indexing %d text chunks...", len(documents_or_chunks))
        vector_store = Chroma.from_texts(
            documents_or_chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

    logger.info("Chroma vector store successfully saved to %s.", persist_directory)
    return vector_store


def load_vector_store(
    persist_directory: str = DEFAULT_VECTOR_DB_PATH,
) -> Chroma:
    """Load an existing ChromaDB vector store from disk.

    Args:
        persist_directory: Local directory path where Chroma database was saved.

    Returns:
        Loaded :class:`Chroma` database instance.
    """
    logger.info("Loading Chroma vector store from %s ...", persist_directory)
    embeddings = get_dense_embeddings()
    return Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
