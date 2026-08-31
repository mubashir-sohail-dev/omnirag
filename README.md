# Custom RAG — Open-Source Dense Retrieval Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A modular, production-ready Python framework for building **Dense Retrieval-Augmented Generation (RAG)** systems using **ChromaDB**, **HuggingFace Dense Embeddings**, and multi-provider LLM integrations (**Google Gemini**, **Groq**, and **Ollama**).

---

## 🌟 Overview

**Custom RAG** turns any collection of PDF documents, scanned files, or text notes into an intelligent, queryable document knowledge base. It includes an automated OCR pipeline, page-aware text splitting, persistent vector indexing, and plug-and-play LLM answer generation with strict anti-hallucination prompts.

---

## ✨ Features

- **Generic Multi-Format Ingestion**: Ingest single PDF files, text documents, or entire directory trees (`.pdf`, `.txt`).
- **Automated OCR Pipeline**: Built-in PyOCR / Tesseract support for scanning image-based PDFs seamlessly.
- **Page-Aware Text Chunking**: Context-preserving recursive splitting with metadata tracking and configurable overlap.
- **Dense Vector Search**: High-performance semantic similarity matching using ChromaDB and `sentence-transformers/all-MiniLM-L6-v2`.
- **Multi-Provider LLM Integration**: Dynamically switch between **Google Gemini**, **Groq (Llama 3.1)**, and local **Ollama** models via simple CLI flags.
- **Production Architecture**: 100% self-contained codebase with zero external project dependencies, comprehensive type hints, logging, and unit tests.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Input Documents: PDF / TXT / Directory] --> B[Ingestion Loader]
    B --> C{File Type?}
    C -->|PDF| D[PyOCR / Page Renderer]
    C -->|TXT| E[Text Reader]
    D --> F[LangChain Documents]
    E --> F
    F --> G[Recursive Text Splitter]
    G --> H[HuggingFace Dense Embeddings]
    H --> I[(ChromaDB Vector Store)]
    
    J[User Query] --> K[Dense Similarity Retriever]
    I --> K
    K --> L[RAG Answer Chain]
    M[LLM Provider: Google / Groq / Ollama] --> L
    L --> N[Grounded Answer Output]
```

---

## 📁 Repository Structure

```text
custom_rag/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration & hyperparameter defaults
├── chunking/
│   ├── __init__.py
│   └── text_splitter.py     # Recursive character text splitting with metadata preservation
├── embeddings/
│   ├── __init__.py
│   └── dense_embeddings.py  # HuggingFace dense embeddings initializer
├── ingestion/
│   ├── __init__.py
│   └── document_loader.py   # Unified PDF, OCR fallback, and TXT loader
├── llm/
│   ├── __init__.py
│   ├── prompts.py           # Anti-hallucination system & RAG chat prompts
│   └── providers.py         # Unified LLM provider factory (Google, Groq, Ollama)
├── retrieval/
│   ├── __init__.py
│   └── rag_chain.py         # Dense similarity retriever & question-answering chain
├── vectorstore/
│   ├── __init__.py
│   └── chroma_store.py      # ChromaDB persistence, creation, and loading
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Logging configuration & recursive file discovery
├── tests/                   # Pytest automated test suite
│   ├── conftest.py          # Shared fixtures
│   ├── test_chunking.py     # Chunking and splitting tests
│   ├── test_generation.py   # Prompt and provider factory tests
│   ├── test_ingestion.py    # Document loading and file discovery tests
│   └── test_vector_store.py # Vector store validation tests
├── rag.py                   # Command-line interface entry point
├── requirements.txt         # Production dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Ignored files, virtual environments, and caches
├── CONTRIBUTING.md          # Contribution guidelines
└── LICENSE                  # MIT License
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
cd custom_rag
pip install -r requirements.txt
```

*Note: Ensure Tesseract OCR and Poppler (for `pdf2image`) are installed on your system path if processing scanned PDFs.*

### 2. Environment Setup

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Example `.env` contents:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## 💻 CLI Usage

### Ingesting Documents

You can ingest single files or entire folders containing `.pdf` and `.txt` files:

#### Single PDF Document
```bash
python rag.py ingest path/to/document.pdf
```

#### Single Text File
```bash
python rag.py ingest path/to/notes.txt
```

#### Entire Directory Tree
```bash
python rag.py ingest ./knowledge_base
```

#### Custom Chunking Parameters
```bash
python rag.py ingest ./knowledge_base --chunk-size 800 --chunk-overlap 150 --vector-db-path ./custom_db
```

---

### Querying the Document Store

Start an interactive question-answering CLI session:

#### Google Gemini (Default)
```bash
python rag.py query --provider google
```

#### Groq (Llama 3.1 8B)
```bash
python rag.py query --provider groq
```

#### Local Ollama (Offline / Local)
```bash
python rag.py query --provider ollama --model llama3.2
```

#### Custom Retrieval Parameters
```bash
python rag.py query --provider google --k 7 --vector-db-path ./custom_db
```

---

## ⚙️ Configuration

All defaults are defined in `config/settings.py` and can be overridden via CLI flags or environment variables:

| Setting | Default Value | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace dense model |
| `DEFAULT_CHUNK_SIZE` | `1000` | Maximum character length per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `DEFAULT_VECTOR_DB_PATH` | `./vector_store` | Local ChromaDB database path |
| `DEFAULT_PROVIDER` | `google` | Default LLM provider (`google`, `groq`, `ollama`) |
| `DEFAULT_TOP_K` | `5` | Chunks retrieved per query |

---

## 🧪 Testing

Run the full pytest suite:

```bash
python -m pytest tests/ -v
```

---

## 🛠️ Troubleshooting

- **OCR Tool Not Found Error**: Install Tesseract OCR (`apt-get install tesseract-ocr` on Linux, `brew install tesseract` on macOS, or the Tesseract installer on Windows) and verify `tesseract --version` works in your terminal.
- **pdf2image / Poppler Error**: Install Poppler binaries (`apt-get install poppler-utils` on Linux, `brew install poppler` on macOS).
- **ChromaDB SQLite Warning**: Upgrade `chromadb` or ensure SQLite 3.35+ is available in your Python environment.

---

## 🗺️ Roadmap

- [ ] Support for Markdown (`.md`), Docx (`.docx`), and HTML ingestion loaders.
- [ ] Hybrid BM25 + Dense ensemble retrieval support.
- [ ] FastApi REST endpoint for serving queries as a web API.
- [ ] Docker containerization.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
