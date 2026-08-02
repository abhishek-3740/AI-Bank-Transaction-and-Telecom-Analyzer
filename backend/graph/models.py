"""Pydantic models for the graph analytics backend module."""

from pydantic import BaseModel
from typing import Optional


class GraphNode(BaseModel):
    """A node in the financial transfer graph."""

    node_id: str
    total_sent: float
    total_received: float
    out_degree: int
    in_degree: int
    in_out_ratio: float
    total_amount: float
    max_risk_score: float
    alert_count: int
    pagerank: float
    suspicion_score: float
    is_mule_account: int


class GraphEdge(BaseModel):
    """A directed edge in the financial transfer graph (one bank transaction)."""

    Transaction_ID: str
    src: str
    dst: str
    Transaction_Amount: float
    risk_score: float
    risk_band: str
    is_suspicious_gt: int


class GraphSummary(BaseModel):
    """High-level graph statistics."""

    total_nodes: int
    total_edges: int
    known_mule_nodes: int
    top_suspicious_nodes: list[GraphNode]
