"""Utility functions for the PDF parser backend module."""

import os
import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile

from .config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE, UPLOAD_DIRECTORY
from .logging_config import get_logger

logger = get_logger(__name__)


def validate_file_extension(filename: str) -> None:
    """Validate that the uploaded file has an allowed extension.

    Args:
        filename: Name of the uploaded file.

    Raises:
        HTTPException: If the file extension is not allowed.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(
            f"Rejected upload: invalid extension '{ext}' for file '{filename}'"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Only PDF files are allowed.",
        )


def validate_file_size(file_size: int) -> None:
    """Validate that the uploaded file does not exceed the maximum size.

    Args:
        file_size: Size of the uploaded file in bytes.

    Raises:
        HTTPException: If the file exceeds the maximum allowed size.
    """
    if file_size > MAX_UPLOAD_SIZE:
        logger.warning(
            f"Rejected upload: file size {file_size} exceeds limit {MAX_UPLOAD_SIZE}"
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_SIZE / (1024 * 1024):.0f} MB.",
        )


def ensure_upload_directory() -> None:
    """Create the upload directory if it does not exist."""
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


def save_upload_file(upload_file: UploadFile, filename: str) -> Path:
    """Save an uploaded file to the upload directory.

    Args:
        upload_file: The uploaded file object.
        filename: Name to save the file as.

    Returns:
        Path to the saved file.
    """
    ensure_upload_directory()
    file_path = UPLOAD_DIRECTORY / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    logger.info(f"Saved uploaded file to {file_path}")
    return file_path


def delete_file(file_path: Path) -> None:
    """Delete a file if it exists.

    Args:
        file_path: Path to the file to delete.
    """
    try:
        if file_path.exists():
            os.remove(file_path)
            logger.info(f"Deleted temporary file {file_path}")
    except OSError as exc:
        logger.error(f"Failed to delete temporary file {file_path}: {exc}")