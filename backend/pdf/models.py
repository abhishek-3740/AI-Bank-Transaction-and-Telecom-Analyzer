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
    # What ingestion did with the parsed rows: how many landed in the upload
    # corpus, whether the dashboard was rebuilt, and what had to be inferred.
    # None when ingestion could not run — parsing still succeeded.
    ingest: dict[str, Any] | None = None


class ParserError(BaseModel):
    """Response model for parser errors."""

    detail: str