"""STR (Suspicious Transaction Report) export endpoints — Stage 12.

Generates forensic investigation reports over the scored transactions and
ground-truth labels, combining ML risk scores, rule engine signals, and graph
analytics into a structured report per customer.

Endpoints
---------
GET  /api/v1/reports/str/{customer_id}     — generate STR for one customer
GET  /api/v1/reports/str/batch             — batch STR for all alerted customers
GET  /api/v1/reports/summary               — portfolio-level summary
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from .models import STRReport, STRTransaction

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# backend/reports/router.py: parents[1]=backend/  parents[2]=repo root
_BACKEND = Path(__file__).resolve().parents[1]   # backend/
_REPO    = Path(__file__).resolve().parents[2]   # TRI-NETRA/ (data/ lives here)
_OUT     = _BACKEND / "notebook" / "output"

_SCORED_CSV = _OUT / "scored_transactions.csv"
_GRAPH_CSV  = _OUT / "graph_analytics.csv"
_GT_CSV     = _REPO / "data" / "ground_truth" / "anomaly_ground_truth.csv"

_scored_df: Optional[pd.DataFrame] = None
_graph_df:  Optional[pd.DataFrame] = None
_gt_df:     Optional[pd.DataFrame] = None


def _load_scored() -> pd.DataFrame:
    global _scored_df
    if _scored_df is not None:
        return _scored_df
    if not _SCORED_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail="Scored transactions not found. Run: python scripts/score.py",
        )
    df = pd.read_csv(_SCORED_CSV, dtype={"Sender_Customer_ID": str,
                                          "Receiver_Account_Number": str,
                                          "Transaction_ID": str})
    df["risk_score"] = df["risk_score"].fillna(0)
    df["rules_fired"] = df["rules_fired"].fillna("")
    for col in ("reason_1", "reason_2", "reason_3"):
        df[col] = df[col].fillna("")
    _scored_df = df
    return df


def _load_graph() -> Optional[pd.DataFrame]:
    global _graph_df
    if _graph_df is not None:
        return _graph_df
    if not _GRAPH_CSV.exists():
        return None
    _graph_df = pd.read_csv(_GRAPH_CSV, dtype={"node_id": str}).fillna(0)
    return _graph_df


def _load_gt() -> Optional[pd.DataFrame]:
    global _gt_df
    if _gt_df is not None:
        return _gt_df
    if not _GT_CSV.exists():
        return None
    _gt_df = pd.read_csv(_GT_CSV, dtype={"Customer_ID": str, "Transaction_ID": str})
    return _gt_df


def _build_str_report(customer_id: str, min_risk: float,
                       officer: str) -> STRReport:
    scored = _load_scored()
    cdf = scored[scored["Sender_Customer_ID"] == customer_id]
    if cdf.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")

    alerts = cdf[cdf["risk_score"] >= min_risk].sort_values("risk_score", ascending=False)
    if alerts.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No transactions at risk >= {min_risk} for customer '{customer_id}'.",
        )

    # Ground truth scenario types if available
    gt = _load_gt()
    scenario_types: list[str] = []
    if gt is not None:
        cgt = gt[gt["Transaction_ID"].isin(set(alerts["Transaction_ID"]))]
        scenario_types = sorted(cgt["Scenario_Type"].dropna().unique().tolist())

    # Graph context if available
    graph = _load_graph()
    graph_suspicion: Optional[float] = None
    graph_ratio: Optional[float] = None
    graph_mule: Optional[int] = None
    if graph is not None:
        sender_acct = alerts["Sender_Customer_ID"].iloc[0]
        g_row = graph[graph["node_id"] == sender_acct]
        if not g_row.empty:
            graph_suspicion = float(g_row.iloc[0].get("suspicion_score", 0))
            graph_ratio     = float(g_row.iloc[0].get("in_out_ratio", 0))
            graph_mule      = int(g_row.iloc[0].get("is_mule_account", 0))

    customer_name = str(alerts["Sender_Customer_Name"].iloc[0])
    total_amount  = float(alerts["Transaction_Amount"].sum())
    primary_band  = str(alerts.iloc[0]["risk_band"])

    txn_list: list[STRTransaction] = []
    for _, row in alerts.iterrows():
        reasons = [r for r in [row["reason_1"], row["reason_2"], row["reason_3"]] if r]
        rules   = [r for r in str(row["rules_fired"]).split("|") if r]
        txn_list.append(STRTransaction(
            Transaction_ID=str(row["Transaction_ID"]),
            Date=str(row["Date"]),
            Timestamp=str(row["Timestamp"]),
            Transaction_Amount=float(row["Transaction_Amount"]),
            risk_score=float(row["risk_score"]),
            risk_band=str(row["risk_band"]),
            reasons=reasons,
            rules_fired=rules,
        ))

    # Narrative auto-generated from signals
    narrative_parts = [
        f"Customer {customer_name} (ID: {customer_id}) has {len(alerts)} transaction(s) "
        f"flagged above risk threshold {min_risk}, totalling INR {total_amount:,.2f}.",
    ]
    if scenario_types:
        narrative_parts.append(
            f"Detected scenario types: {', '.join(scenario_types)}."
        )
    if graph_suspicion is not None:
        narrative_parts.append(
            f"Graph analysis: suspicion score {graph_suspicion:.4f}, "
            f"in/out ratio {graph_ratio:.2f}."
            + (" Node flagged as potential mule account." if graph_mule else "")
        )
    fired_rules: list[str] = []
    for r_str in alerts["rules_fired"]:
        fired_rules.extend([r for r in str(r_str).split("|") if r])
    if fired_rules:
        from collections import Counter
        rule_summary = ", ".join(
            f"{r} ×{c}" for r, c in Counter(fired_rules).most_common()
        )
        narrative_parts.append(f"Rule engine signals: {rule_summary}.")
    narrative_parts.append(
        "This report is system-generated and must be reviewed by a certified "
        "reporting officer before submission."
    )

    return STRReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc).isoformat(),
        customer_id=customer_id,
        customer_name=customer_name,
        reporting_officer=officer,
        total_suspicious_transactions=len(alerts),
        total_suspicious_amount=round(total_amount, 2),
        date_range_from=str(alerts["Date"].min()),
        date_range_to=str(alerts["Date"].max()),
        primary_risk_band=primary_band,
        scenario_types_detected=scenario_types,
        transactions=txn_list,
        narrative=" ".join(narrative_parts),
        graph_suspicion_score=graph_suspicion,
        graph_in_out_ratio=graph_ratio,
        graph_mule_flag=graph_mule,
    )


@router.get("/str/batch", response_model=list[dict])
def batch_str_summary(
    min_risk: float = Query(70.0, ge=0, le=100),
    top_n: int = Query(20, ge=1, le=200, description="Max customers to return"),
) -> list[dict]:
    """Return a lightweight STR summary for the top-N alerted customers.

    Returns customer_id, name, alert_count, total_amount, primary_risk_band.
    Use GET /str/{customer_id} to fetch the full report for any individual.
    """
    scored = _load_scored()
    alerts = scored[scored["risk_score"] >= min_risk]
    agg = (alerts.groupby("Sender_Customer_ID")
           .agg(
               customer_name=("Sender_Customer_Name", "first"),
               alert_count=("Transaction_ID", "count"),
               total_amount=("Transaction_Amount", "sum"),
               max_risk=("risk_score", "max"),
           )
           .reset_index()
           .sort_values("max_risk", ascending=False)
           .head(top_n))
    return agg.rename(columns={"Sender_Customer_ID": "customer_id"}).to_dict(orient="records")


@router.get("/str/{customer_id}", response_model=STRReport)
def generate_str(
    customer_id: str,
    min_risk: float = Query(70.0, ge=0, le=100),
    officer: str = Query("System", description="Name of the reporting officer"),
) -> STRReport:
    """Generate a Suspicious Transaction Report for a single customer."""
    return _build_str_report(customer_id, min_risk, officer)


@router.get("/summary")
def portfolio_summary(
    min_risk: float = Query(70.0, ge=0, le=100),
) -> dict:
    """Portfolio-level investigation summary — total customers, amounts, breakdown."""
    scored = _load_scored()
    alerts = scored[scored["risk_score"] >= min_risk]
    unique_customers = alerts["Sender_Customer_ID"].nunique()
    band_dist = alerts["risk_band"].value_counts().to_dict()

    rule_counts: dict[str, int] = {}
    for r_str in alerts["rules_fired"].dropna():
        for r in str(r_str).split("|"):
            if r:
                rule_counts[r] = rule_counts.get(r, 0) + 1

    return {
        "total_alerts": len(alerts),
        "unique_alerted_customers": unique_customers,
        "total_suspicious_amount_inr": round(float(alerts["Transaction_Amount"].sum()), 2),
        "band_distribution": band_dist,
        "rule_breakdown": rule_counts,
        "min_risk_threshold": min_risk,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
