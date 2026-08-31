"""Centralized configuration settings for custom_rag."""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
LOG_LEVEL: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
"""Default logging level for application logs."""

# Embedding Model Configuration
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
"""HuggingFace model used for dense embeddings in ChromaDB."""

# Chunking Configuration
DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
"""Default maximum character count per text chunk."""

DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
"""Default overlapping characters between adjacent chunks."""

CHUNK_SEPARATORS: list[str] = ["\n[PAGE", "\n\n", "\n", ". ", " ", ""]
"""Ordered list of separators for recursive character text splitting."""

# Vector Store Configuration
DEFAULT_VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_store")
"""Default local path for persisting Chroma database."""

DEFAULT_COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "custom_rag_collection")
"""Default collection name in ChromaDB."""

# Retrieval Configuration
DEFAULT_TOP_K: int = int(os.getenv("TOP_K", "5"))
"""Default number of document chunks to retrieve per query."""

# Ingestion & OCR Configuration
SUPPORTED_FILE_EXTENSIONS: set[str] = {".pdf", ".txt"}
"""Supported document extensions for ingestion."""

OCR_MIN_TEXT_LENGTH: int = int(os.getenv("OCR_MIN_TEXT_LENGTH", "50"))
"""Minimum extracted characters per PDF page to consider native text valid (below this triggers OCR fallback)."""

OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
"""Language code for PyOCR / Tesseract extraction."""

# LLM Provider Configuration
DEFAULT_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "google").lower()
"""Default LLM provider ('google', 'groq', 'ollama')."""

DEFAULT_MODEL_MAP: dict[str, str] = {
    "google": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
    "groq": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
}
"""Default LLM model name map per provider."""

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
"""Base URL for local or remote Ollama server."""
