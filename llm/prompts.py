"""System and RAG prompt definitions for custom_rag."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT: str = (
    "You are a helpful AI assistant specializing in document question answering. "
    "Your ONLY job is to extract answers directly from the provided context. "
    "If the provided context does not contain the exact answer, you MUST reply "
    "with: 'I cannot answer this based on the retrieved documents.' "
    "Under NO circumstances should you guess, use outside knowledge, or make "
    "up an answer.\n\n"
    "Context: {context}"
)
"""System prompt used across retrieval chains."""

RAG_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{input}"),
])
"""Chat prompt template combining system rules and user input."""
