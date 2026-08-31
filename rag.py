"""Custom RAG — Standalone Open-Source Dense Retrieval Framework CLI.

Provides ``ingest`` and ``query`` commands for building vector stores
from documents (PDFs, TXT files, or directories) and interactively querying
with configurable LLM providers (Google, Groq, Ollama).
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Ensure local package imports resolve when executed directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunking.text_splitter import split_documents
from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PROVIDER,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_DB_PATH,
)
from ingestion.document_loader import load_documents
from retrieval.rag_chain import generate_answer
from utils.helpers import setup_logging
from vectorstore.chroma_store import create_vector_store

load_dotenv()
logger = logging.getLogger(__name__)


def ingest(args: argparse.Namespace) -> None:
    """Ingest single files or directories into ChromaDB vector store.

    Args:
        args: Parsed command line arguments.
    """
    input_path = args.input_path

    print("\nLoading documents...")
    try:
        docs = load_documents(input_path)
    except (FileNotFoundError, ValueError) as err:
        print(f"❌ Ingestion error: {err}")
        sys.exit(1)

    if not docs:
        print("❌ Error: No valid document pages were extracted.")
        sys.exit(1)

    print("\nChunking documents...")
    chunked_docs = split_documents(
        docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    print(f"Created {len(chunked_docs)} chunks.")

    print("\nGenerating embeddings...")
    print("Persisting Chroma database...")
    create_vector_store(chunked_docs, persist_directory=args.vector_db_path)

    print("\nDone. Ingestion completed successfully! ✨\n")


def query(args: argparse.Namespace) -> None:
    """Run interactive question-answering session.

    Args:
        args: Parsed command line arguments.
    """
    print("\n📚 Custom RAG Framework — type 'exit' to quit.\n")
    print(f"Provider: {args.provider} | Vector DB: {args.vector_db_path}\n")

    while True:
        try:
            user_query = input("Enter your question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_query.strip().lower() == "exit":
            print("Goodbye!")
            break

        if not user_query.strip():
            continue

        try:
            answer = generate_answer(
                user_query,
                provider=args.provider,
                model_name=args.model,
                persist_directory=args.vector_db_path,
                k=args.k,
            )
            print(f"\nAnswer 🤖:\n{answer}\n")
        except Exception as err:
            print(f"\n❌ Query error: {err}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="custom_rag",
        description="Custom Open-Source Dense Retrieval RAG Framework.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest PDF/TXT files or directories into ChromaDB vector store",
    )
    ingest_parser.add_argument(
        "input_path",
        help="Path to input PDF file, text file, or directory containing documents",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum characters per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlapping characters between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    ingest_parser.add_argument(
        "--vector-db-path",
        "--chroma-dir",
        dest="vector_db_path",
        default=DEFAULT_VECTOR_DB_PATH,
        help=f"Path to persist ChromaDB (default: {DEFAULT_VECTOR_DB_PATH})",
    )

    # Query subcommand
    query_parser = subparsers.add_parser(
        "query", help="Interactively ask questions against the vector store"
    )
    query_parser.add_argument(
        "--provider",
        choices=["google", "groq", "ollama"],
        default=DEFAULT_PROVIDER,
        help=f"LLM provider to use (default: {DEFAULT_PROVIDER})",
    )
    query_parser.add_argument(
        "--model",
        default=None,
        help="Optional model name override for the selected provider",
    )
    query_parser.add_argument(
        "--vector-db-path",
        "--chroma-dir",
        dest="vector_db_path",
        default=DEFAULT_VECTOR_DB_PATH,
        help=f"Path to ChromaDB vector store (default: {DEFAULT_VECTOR_DB_PATH})",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of document chunks to retrieve (default: {DEFAULT_TOP_K})",
    )

    return parser


def main() -> None:
    """Main CLI entry point."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args)
    elif args.command == "query":
        query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
