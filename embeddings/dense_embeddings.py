"""Dense embedding generation module using HuggingFace."""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_dense_embeddings(
    model_name: str = EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Initialize and return a HuggingFace dense embedding model instance.

    Args:
        model_name: HuggingFace model name.

    Returns:
        Configured :class:`HuggingFaceEmbeddings` object.
    """
    logger.info("Initializing HuggingFace dense embeddings: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)
