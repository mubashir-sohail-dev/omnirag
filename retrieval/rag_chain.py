"""RAG retrieval and answer generation chain module."""

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from config.settings import (
    DEFAULT_PROVIDER,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_DB_PATH,
)
from llm.prompts import RAG_PROMPT
from llm.providers import get_llm
from vectorstore.chroma_store import load_vector_store

logger = logging.getLogger(__name__)


def get_dense_retriever(vector_store: Chroma, k: int = DEFAULT_TOP_K):
    """Create and return a dense similarity retriever from Chroma.

    Args:
        vector_store: Instantiated Chroma vector store.
        k: Number of top document chunks to retrieve.

    Returns:
        Vector store retriever object.
    """
    logger.info("Building dense similarity retriever (k=%d)...", k)
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )


def generate_answer(
    query: str,
    provider: str = DEFAULT_PROVIDER,
    model_name: Optional[str] = None,
    persist_directory: str = DEFAULT_VECTOR_DB_PATH,
    k: int = DEFAULT_TOP_K,
) -> str:
    """Generate an answer to a query using dense retrieval and selected LLM provider.

    Args:
        query: User question string.
        provider: LLM provider name ('google', 'groq', 'ollama').
        model_name: Optional model identifier override.
        persist_directory: Path to Chroma database.
        k: Top-k document chunks to retrieve.

    Returns:
        Generated answer string.
    """
    logger.info("Retrieving relevant context from Chroma database...")
    db = load_vector_store(persist_directory=persist_directory)
    retriever = get_dense_retriever(db, k=k)

    logger.info("Initializing LLM provider '%s'...", provider)
    llm = get_llm(provider=provider, model_name=model_name)

    question_answer_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    logger.info("Running query against RAG chain...")
    response = rag_chain.invoke({"input": query})

    answer: str = response["answer"]
    logger.info("Answer generated successfully.")
    return answer
