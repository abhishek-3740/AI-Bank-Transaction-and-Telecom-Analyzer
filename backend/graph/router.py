"""Graph analytics endpoints — Stage 9.

Serves the node and edge tables written by scripts/graph_analytics.py.

Endpoints
---------
GET /api/v1/graph/summary          — overall graph statistics
GET /api/v1/graph/nodes            — paginated node list, sortable
GET /api/v1/graph/node/{node_id}   — single node detail + its edges
GET /api/v1/graph/mules            — nodes flagged as likely mule accounts
GET /api/v1/graph/edges            — paginated edge list
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from csv_cache import load_csv_cached

from .models import GraphEdge, GraphNode, GraphSummary

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])

# backend/graph/router.py: parents[1] = backend/
_BACKEND   = Path(__file__).resolve().parents[1]
_OUT       = _BACKEND / "notebook" / "output"
_NODES_CSV = _OUT / "graph_analytics.csv"
_EDGES_CSV = _OUT / "graph_edges.csv"

def _load_nodes() -> pd.DataFrame:
    if not _NODES_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail="No graph analytics yet. Upload a bank statement PDF via "
                   "/api/v1/pdf/parse, or run: python scripts/graph_analytics.py",
        )
    return load_csv_cached(
        _NODES_CSV,
        lambda p: pd.read_csv(p, dtype={"node_id": str}).fillna(0),
    )


def _load_edges() -> pd.DataFrame:
    if not _EDGES_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail="No graph edges yet. Upload a bank statement PDF via "
                   "/api/v1/pdf/parse, or run: python scripts/graph_analytics.py",
        )
    return load_csv_cached(
        _EDGES_CSV,
        lambda p: pd.read_csv(p, dtype={"src": str, "dst": str,
                                        "Transaction_ID": str}).fillna(0),
    )


def _row_to_node(row: pd.Series) -> GraphNode:
    return GraphNode(
        node_id=str(row.node_id),
        total_sent=float(row.get("total_sent", 0)),
        total_received=float(row.get("total_received", 0)),
        out_degree=int(row.get("out_degree", 0)),
        in_degree=int(row.get("in_degree", 0)),
        in_out_ratio=float(row.get("in_out_ratio", 0)),
        total_amount=float(row.get("total_amount", 0)),
        max_risk_score=float(row.get("max_risk_score", 0)),
        alert_count=int(row.get("alert_count", 0)),
        pagerank=float(row.get("pagerank", 0)),
        suspicion_score=float(row.get("suspicion_score", 0)),
        is_mule_account=int(row.get("is_mule_account", 0)),
    )


def _row_to_edge(row: pd.Series) -> GraphEdge:
    return GraphEdge(
        Transaction_ID=str(row.Transaction_ID),
        src=str(row.src),
        dst=str(row.dst),
        Transaction_Amount=float(row.Transaction_Amount),
        risk_score=float(row.get("risk_score", 0)),
        risk_band=str(row.get("risk_band", "LOW")),
        is_suspicious_gt=int(row.get("is_suspicious_gt", 0)),
    )


@router.get("/summary", response_model=GraphSummary)
def get_graph_summary(top_n: int = Query(10, ge=1, le=100)) -> GraphSummary:
    """Return overall graph statistics and the top-N suspicious nodes."""
    nodes = _load_nodes()
    top = nodes.nlargest(top_n, "suspicion_score")
    return GraphSummary(
        total_nodes=len(nodes),
        total_edges=len(_load_edges()),
        known_mule_nodes=int(nodes["is_mule_account"].sum()),
        top_suspicious_nodes=[_row_to_node(r) for _, r in top.iterrows()],
    )


@router.get("/nodes", response_model=list[GraphNode])
def get_nodes(
    sort_by: str = Query("suspicion_score",
                         description="Column to sort by: suspicion_score|pagerank|in_out_ratio|alert_count"),
    min_suspicion: float = Query(0.0, ge=0, le=1),
    mule_only: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> list[GraphNode]:
    """Return the node list, sorted and optionally filtered."""
    nodes = _load_nodes()
    mask = nodes["suspicion_score"] >= min_suspicion
    if mule_only:
        mask &= nodes["is_mule_account"] == 1
    valid_sort = {"suspicion_score", "pagerank", "in_out_ratio", "alert_count",
                  "in_degree", "total_received"}
    col = sort_by if sort_by in valid_sort else "suspicion_score"
    filtered = nodes[mask].sort_values(col, ascending=False)
    start = (page - 1) * page_size
    return [_row_to_node(r) for _, r in filtered.iloc[start: start + page_size].iterrows()]


@router.get("/node/{node_id}", response_model=dict)
def get_node_detail(node_id: str) -> dict:
    """Return a single node with all its outgoing and incoming edges."""
    nodes = _load_nodes()
    edges = _load_edges()
    node_rows = nodes[nodes["node_id"] == node_id]
    if node_rows.empty:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    node = _row_to_node(node_rows.iloc[0])
    outgoing = edges[edges["src"] == node_id]
    incoming = edges[edges["dst"] == node_id]
    return {
        "node": node.model_dump(),
        "outgoing_edges": [_row_to_edge(r).model_dump() for _, r in outgoing.iterrows()],
        "incoming_edges": [_row_to_edge(r).model_dump() for _, r in incoming.iterrows()],
    }


@router.get("/mules", response_model=list[GraphNode])
def get_mule_accounts() -> list[GraphNode]:
    """Return all nodes flagged as known or suspected mule accounts, sorted by suspicion."""
    nodes = _load_nodes()
    mules = nodes[nodes["is_mule_account"] == 1].sort_values("suspicion_score", ascending=False)
    return [_row_to_node(r) for _, r in mules.iterrows()]


@router.get("/edges", response_model=list[GraphEdge])
def get_edges(
    src: Optional[str] = Query(None, description="Filter by source (sender) node_id"),
    dst: Optional[str] = Query(None, description="Filter by destination (receiver) node_id"),
    min_risk: float = Query(0.0, ge=0, le=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
) -> list[GraphEdge]:
    """Return edges (transactions), filterable by source/destination/risk."""
    edges = _load_edges()
    mask = edges["risk_score"] >= min_risk
    if src:
        mask &= edges["src"] == src
    if dst:
        mask &= edges["dst"] == dst
    filtered = edges[mask].sort_values("risk_score", ascending=False)
    start = (page - 1) * page_size
    return [_row_to_edge(r) for _, r in filtered.iloc[start: start + page_size].iterrows()]
