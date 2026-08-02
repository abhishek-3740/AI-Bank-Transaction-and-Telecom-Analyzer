"""Pydantic models for the STR (Suspicious Transaction Report) module."""

from typing import Any, Optional
from pydantic import BaseModel


class STRTransaction(BaseModel):
    """One transaction line item in an STR."""

    Transaction_ID: str
    Date: str
    Timestamp: str
    Transaction_Amount: float
    risk_score: float
    risk_band: str
    reasons: list[str]
    rules_fired: list[str]


class STRReport(BaseModel):
    """A Suspicious Transaction Report covering one customer investigation."""

    report_id: str
    generated_at: str
    customer_id: str
    customer_name: str
    reporting_officer: str
    total_suspicious_transactions: int
    total_suspicious_amount: float
    date_range_from: str
    date_range_to: str
    primary_risk_band: str
    scenario_types_detected: list[str]
    transactions: list[STRTransaction]
    narrative: str
    graph_suspicion_score: Optional[float] = None
    graph_in_out_ratio: Optional[float] = None
    graph_mule_flag: Optional[int] = None
