"""Utility helper functions for logging and file path discovery."""

import logging
import os
from config.settings import LOG_LEVEL, SUPPORTED_FILE_EXTENSIONS


def setup_logging(level: int = LOG_LEVEL) -> None:
    """Configure standardized logging output across custom_rag modules."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def validate_path_exists(path: str) -> None:
    """Validate that a file or directory path exists.

    Args:
        path: Path to validate.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: '{path}'")


def discover_supported_files(path: str) -> list[str]:
    """Discover all supported document files (.pdf, .txt) at a path.

    If *path* is a single file, validates its extension.
    If *path* is a directory, recursively walks all subdirectories.

    Args:
        path: Single file path or directory path.

    Returns:
        Sorted list of document file paths.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If file extension is unsupported or directory contains no
            supported documents.
    """
    validate_path_exists(path)

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported extensions: "
                f"{', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
            )
        return [os.path.abspath(path)]

    discovered: list[str] = []
    for root, _, files in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_FILE_EXTENSIONS:
                discovered.append(os.path.abspath(os.path.join(root, file)))

    if not discovered:
        raise ValueError(
            f"No supported document files ({', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}) "
            f"were found in directory: '{path}'"
        )

    return sorted(discovered)
