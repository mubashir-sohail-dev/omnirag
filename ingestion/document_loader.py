"""Unified document ingestion loader supporting native PDF extraction, automatic OCR fallback, and directory walking."""

import logging
import os
from typing import Optional

import pyocr
import pyocr.builders
from langchain_core.documents import Document
from pdf2image import convert_from_path
from pypdf import PdfReader

from config.settings import OCR_MIN_TEXT_LENGTH
from utils.helpers import discover_supported_files, validate_path_exists

logger = logging.getLogger(__name__)


def _run_ocr_on_page(page_image) -> str:
    """Run PyOCR text extraction on a single page image object.

    Args:
        page_image: PIL Image object of the PDF page.

    Returns:
        Extracted text string.

    Raises:
        RuntimeError: If Tesseract or PyOCR is not installed.
    """
    tools = pyocr.get_available_tools()
    if not tools:
        raise RuntimeError(
            "No OCR tool found. Please install Tesseract-OCR and ensure it is in PATH."
        )
    tool = tools[0]
    return tool.image_to_string(
        page_image, lang="eng", builder=pyocr.builders.TextBuilder()
    )


def load_txt_file(file_path: str) -> list[Document]:
    """Load a plain text file into a LangChain Document with metadata.

    Args:
        file_path: Path to text file.

    Returns:
        List containing a single :class:`Document`.
    """
    validate_path_exists(file_path)
    logger.info("Loading TXT file: %s", file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    metadata = {
        "source": os.path.abspath(file_path),
        "filename": os.path.basename(file_path),
        "page": 1,
        "total_pages": 1,
        "file_type": "txt",
        "ocr_used": False,
    }
    return [Document(page_content=content, metadata=metadata)]


def load_pdf_file(pdf_path: str) -> list[Document]:
    """Load a PDF file trying native text extraction first, with automatic OCR fallback.

    For each page:
    1. Attempts native text extraction using pypdf.
    2. If page text length < OCR_MIN_TEXT_LENGTH, converts page image and runs OCR.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        List of :class:`Document` objects (one per page) with preserved metadata.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
    """
    validate_path_exists(pdf_path)
    logger.info("Loading PDF file: %s", pdf_path)

    abs_path = os.path.abspath(pdf_path)
    filename = os.path.basename(pdf_path)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    documents: list[Document] = []

    rendered_images: Optional[list] = None
    ocr_pages_count = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        native_text = (page.extract_text() or "").strip()
        ocr_used = False

        if len(native_text) >= OCR_MIN_TEXT_LENGTH:
            page_content = f"[PAGE {page_num}]\n" + native_text
        else:
            logger.info(
                "Native text sparse on page %d of %s (%d chars). Executing OCR fallback...",
                page_num,
                filename,
                len(native_text),
            )
            if rendered_images is None:
                rendered_images = convert_from_path(pdf_path)

            ocr_text = _run_ocr_on_page(rendered_images[idx]).strip()
            page_content = f"[PAGE {page_num}]\n" + ocr_text
            ocr_used = True
            ocr_pages_count += 1

        metadata = {
            "source": abs_path,
            "filename": filename,
            "page": page_num,
            "total_pages": total_pages,
            "file_type": "pdf",
            "ocr_used": ocr_used,
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    if ocr_pages_count > 0:
        logger.info(
            "PDF %s processed: %d page(s) natively, %d page(s) via OCR fallback.",
            filename,
            total_pages - ocr_pages_count,
            ocr_pages_count,
        )
    else:
        logger.info("PDF %s processed using 100%% native text extraction.", filename)

    return documents


def load_documents(path: str) -> list[Document]:
    """Load and ingest all supported document files (.pdf, .txt) at a path.

    Supports single files, flat directories, and deeply nested directory trees.

    Args:
        path: File or directory path.

    Returns:
        List of loaded :class:`Document` objects with full metadata preserved.
    """
    target_files = discover_supported_files(path)
    logger.info("Loading documents... Found %d file(s) at '%s'.", len(target_files), path)

    all_documents: list[Document] = []
    for file_path in target_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            docs = load_pdf_file(file_path)
        elif ext == ".txt":
            docs = load_txt_file(file_path)
        else:
            continue
        all_documents.extend(docs)

    logger.info(
        "Ingestion complete. Extracted %d page document(s).", len(all_documents)
    )
    return all_documents
