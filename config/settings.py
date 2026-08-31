"""Centralized configuration settings for custom_rag."""

import logging
import os

# Logging Configuration
LOG_LEVEL: int = logging.INFO
"""Default logging level for application logs."""

# Embedding Model Configuration
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
"""HuggingFace model used for dense embeddings in ChromaDB."""

# Chunking Configuration
DEFAULT_CHUNK_SIZE: int = 1000
"""Default maximum character count per text chunk."""

DEFAULT_CHUNK_OVERLAP: int = 200
"""Default overlapping characters between adjacent chunks."""

CHUNK_SEPARATORS: list[str] = ["\n[PAGE", "\n\n", "\n", ". ", " ", ""]
"""Ordered list of separators for recursive character text splitting."""

# Vector Store Configuration
DEFAULT_VECTOR_DB_PATH: str = "./vector_store"
"""Default local path for persisting Chroma database."""

DEFAULT_COLLECTION_NAME: str = "custom_rag_collection"
"""Default collection name in ChromaDB."""

# Retrieval Configuration
DEFAULT_TOP_K: int = 5
"""Default number of document chunks to retrieve per query."""

# Ingestion & OCR Configuration
SUPPORTED_FILE_EXTENSIONS: set[str] = {".pdf", ".txt"}
"""Supported document extensions for ingestion."""

OCR_MIN_TEXT_LENGTH: int = 50
"""Minimum extracted characters per PDF page to consider native text valid (below this triggers OCR fallback)."""

# LLM Provider Configuration
DEFAULT_PROVIDER: str = "google"
"""Default LLM provider ('google', 'groq', 'ollama')."""

DEFAULT_MODEL_MAP: dict[str, str] = {
    "google": "gemini-1.5-flash",
    "groq": "llama-3.1-8b-instant",
    "ollama": "llama3.2",
}
"""Default LLM model name map per provider."""
