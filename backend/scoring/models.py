"""Pydantic models for the scoring/investigation backend module."""

from typing import Any, Optional
from pydantic import BaseModel


class ScoredTransaction(BaseModel):
    """A single scored transaction returned by the investigation API."""

    Transaction_ID: str
    Date: str
    Timestamp: str
    Sender_Customer_ID: str
    Sender_Customer_Name: str
    Receiver_Account_Number: str
    Transaction_Amount: float
    ml_probability: float
    risk_score: float
    risk_band: str
    reason_1: str
    reason_2: str
    reason_3: str
    rules_fired: str
    split: str
    is_suspicious_gt: int


class AlertListResponse(BaseModel):
    """Paginated list of alerts above the risk threshold."""

    total: int
    page: int
    page_size: int
    results: list[ScoredTransaction]


class CustomerSummary(BaseModel):
    """Aggregated risk profile for a single customer."""

    customer_id: str
    customer_name: str
    total_transactions: int
    alert_count: int
    max_risk_score: float
    dominant_risk_band: str
    rules_fired_summary: list[str]
    top_transactions: list[ScoredTransaction]


class ScoreRequest(BaseModel):
    """Request body for on-demand scoring of a single transaction dict."""

    transaction: dict[str, Any]


class ScoreResponse(BaseModel):
    """Result of on-demand scoring."""

    Transaction_ID: str
    ml_probability: float
    risk_score: float
    risk_band: str
    reasons: list[str]
    rules_fired: list[str]
