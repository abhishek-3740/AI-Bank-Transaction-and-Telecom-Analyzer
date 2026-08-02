"""Pydantic models for the PDF parser backend module."""

from typing import Any

from pydantic import BaseModel


class ParserResponse(BaseModel):
    """Response model for successful PDF parsing."""

    status: str
    dataset_type: str
    rows: int
    columns: list[str]
    data: list[dict[str, Any]]


class ParserError(BaseModel):
    """Response model for parser errors."""

    detail: str