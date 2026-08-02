"""Configuration values for the PDF parser backend module."""

import os
from pathlib import Path

PROJECT_NAME: str = "AI Telecom Bank Analyzer - PDF Parser"
VERSION: str = "1.0.0"

# Upload directory for temporary PDF storage — platform-safe default
import tempfile
_default_upload = str(Path(tempfile.gettempdir()) / "tri-netra-uploads")
UPLOAD_DIRECTORY: Path = Path(os.getenv("UPLOAD_DIRECTORY", _default_upload))

# Maximum upload size in bytes (50 MB)
MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", 50 * 1024 * 1024))

# Allowed file extensions
ALLOWED_EXTENSIONS: set[str] = {".pdf"}