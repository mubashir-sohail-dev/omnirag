# Contributing to Custom RAG

Thank you for your interest in contributing to **Custom RAG**!

## Development Setup

1. Clone the repository and navigate to `custom_rag/`:
   ```bash
   cd custom_rag
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

Run the unit test suite before submitting pull requests:
```bash
python -m pytest tests/ -v
```

## Guidelines
- Follow PEP 8 style guidelines.
- Include Google-style docstrings and type hints for all new functions.
- Write unit tests for new functionality in `tests/`.
