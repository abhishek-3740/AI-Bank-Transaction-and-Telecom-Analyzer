"""Investigation / scoring endpoints — Stage 10.

Serves the pre-scored CSV written by scripts/score.py, with query parameters
for filtering, pagination, and customer drill-down. Also exposes a lightweight
on-demand scoring endpoint for single transactions (uses the persisted model
bundle, never retrains).

Endpoints
---------
GET  /api/v1/scoring/alerts         — all alerts at risk >= threshold
GET  /api/v1/scoring/transactions   — full scored table, filterable
GET  /api/v1/scoring/customer/{id}  — per-customer risk summary
GET  /api/v1/scoring/stats          — aggregate dashboard numbers
POST /api/v1/scoring/score          — score a single raw bank transaction
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from csv_cache import load_csv_cached

from .models import (
    AlertListResponse,
    CustomerSummary,
    ScoreRequest,
    ScoreResponse,
    ScoredTransaction,
)

router = APIRouter(prefix="/api/v1/scoring", tags=["scoring"])

# backend/scoring/router.py  →  parents[1]=backend/  parents[2]=repo root
_BACKEND_DIR = Path(__file__).resolve().parents[1]      # backend/
_REPO_ROOT   = Path(__file__).resolve().parents[2]      # TRI-NETRA/
_SCORED_CSV  = _BACKEND_DIR / "notebook" / "output" / "scored_transactions.csv"
_BUNDLE_PATH = _BACKEND_DIR / "models" / "stage7_setC.joblib"
_SCRIPTS_DIR = _BACKEND_DIR / "scripts"

# ---------------------------------------------------------------------------
# Data loading (lazy, re-read whenever the CSV changes on disk)
# ---------------------------------------------------------------------------

def _read_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Sender_Customer_ID": str,
                                  "Receiver_Account_Number": str,
                                  "Transaction_ID": str})
    df["risk_score"] = df["risk_score"].fillna(0)
    df["is_suspicious_gt"] = df["is_suspicious_gt"].fillna(0).astype(int)
    df["rules_fired"] = df["rules_fired"].fillna("")
    for col in ("reason_1", "reason_2", "reason_3"):
        df[col] = df[col].fillna("")
    return df


def _load_scored() -> pd.DataFrame:
    """Load the scored transactions CSV, raising a clear error if missing."""
    if not _SCORED_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No scored transactions yet. Upload a bank statement PDF via "
                "/api/v1/pdf/parse, or run: python scripts/score.py"
            ),
        )
    return load_csv_cached(_SCORED_CSV, _read_scored)


def _row_to_model(row: pd.Series) -> ScoredTransaction:
    return ScoredTransaction(
        Transaction_ID=str(row.Transaction_ID),
        Date=str(row.Date),
        Timestamp=str(row.Timestamp),
        Sender_Customer_ID=str(row.Sender_Customer_ID),
        Sender_Customer_Name=str(row.Sender_Customer_Name),
        Receiver_Account_Number=str(row.Receiver_Account_Number),
        Transaction_Amount=float(row.Transaction_Amount),
        ml_probability=float(row.ml_probability),
        risk_score=float(row.risk_score),
        risk_band=str(row.risk_band),
        reason_1=str(row.reason_1),
        reason_2=str(row.reason_2),
        reason_3=str(row.reason_3),
        rules_fired=str(row.rules_fired),
        split=str(row.split),
        is_suspicious_gt=int(row.is_suspicious_gt),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=AlertListResponse)
def get_alerts(
    min_risk: float = Query(70.0, ge=0, le=100, description="Minimum risk score"),
    band: Optional[str] = Query(None, description="Filter by band: CRITICAL|HIGH|MEDIUM|LOW"),
    split: Optional[str] = Query(None, description="Filter by data split: train|val|test"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> AlertListResponse:
    """Return all transactions at or above *min_risk*, sorted by risk score descending."""
    df = _load_scored()
    mask = df["risk_score"] >= min_risk
    if band:
        mask &= df["risk_band"].str.upper() == band.upper()
    if split:
        mask &= df["split"] == split
    filtered = df[mask].sort_values("risk_score", ascending=False)
    total = len(filtered)
    start = (page - 1) * page_size
    page_df = filtered.iloc[start : start + page_size]
    return AlertListResponse(
        total=total,
        page=page,
        page_size=page_size,
        results=[_row_to_model(r) for _, r in page_df.iterrows()],
    )


@router.get("/transactions", response_model=AlertListResponse)
def get_transactions(
    customer_id: Optional[str] = Query(None, description="Filter by Sender_Customer_ID"),
    min_risk: float = Query(0.0, ge=0, le=100),
    max_risk: float = Query(100.0, ge=0, le=100),
    date_from: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    rule: Optional[str] = Query(None, description="Filter by rule name, e.g. ODD_HOUR"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
) -> AlertListResponse:
    """Search the full scored transaction table with optional filters."""
    df = _load_scored()
    mask = (df["risk_score"] >= min_risk) & (df["risk_score"] <= max_risk)
    if customer_id:
        mask &= df["Sender_Customer_ID"] == customer_id
    if date_from:
        mask &= df["Date"] >= date_from
    if date_to:
        mask &= df["Date"] <= date_to
    if rule:
        mask &= df["rules_fired"].str.contains(rule, case=False, na=False)
    filtered = df[mask].sort_values("risk_score", ascending=False)
    total = len(filtered)
    start = (page - 1) * page_size
    page_df = filtered.iloc[start : start + page_size]
    return AlertListResponse(
        total=total,
        page=page,
        page_size=page_size,
        results=[_row_to_model(r) for _, r in page_df.iterrows()],
    )


@router.get("/customer/{customer_id}", response_model=CustomerSummary)
def get_customer_summary(
    customer_id: str,
    top_n: int = Query(5, ge=1, le=50, description="Number of top transactions to return"),
) -> CustomerSummary:
    """Return the risk profile for a single customer."""
    df = _load_scored()
    cdf = df[df["Sender_Customer_ID"] == customer_id]
    if cdf.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found.")

    alerts = cdf[cdf["risk_score"] >= 70]
    max_risk = float(cdf["risk_score"].max())

    band_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    dominant_band = (
        cdf.loc[cdf["risk_score"].idxmax(), "risk_band"]
        if max_risk > 0 else "LOW"
    )

    all_rules: list[str] = []
    for rules_str in cdf["rules_fired"].dropna():
        if rules_str:
            all_rules.extend(rules_str.split("|"))
    rules_summary = sorted(set(all_rules))

    top_txns = cdf.nlargest(top_n, "risk_score")
    name = cdf["Sender_Customer_Name"].iloc[0]

    return CustomerSummary(
        customer_id=customer_id,
        customer_name=str(name),
        total_transactions=len(cdf),
        alert_count=len(alerts),
        max_risk_score=max_risk,
        dominant_risk_band=str(dominant_band),
        rules_fired_summary=rules_summary,
        top_transactions=[_row_to_model(r) for _, r in top_txns.iterrows()],
    )


@router.get("/stats")
def get_stats() -> dict:
    """Aggregate numbers for the investigation dashboard."""
    df = _load_scored()
    alerts = df[df["risk_score"] >= 70]
    band_counts = df["risk_band"].value_counts().to_dict()
    rule_counts: dict[str, int] = {}
    for rules_str in df["rules_fired"].dropna():
        for r in rules_str.split("|"):
            if r:
                rule_counts[r] = rule_counts.get(r, 0) + 1

    # Precision / recall only calculable on test split (not in-sample)
    test = df[df["split"] == "test"]
    test_alerts = test[test["risk_score"] >= 70]
    precision = (test_alerts["is_suspicious_gt"].mean()
                 if len(test_alerts) > 0 else None)
    recall = (test_alerts["is_suspicious_gt"].sum() / max(test["is_suspicious_gt"].sum(), 1)
              if len(test_alerts) > 0 else None)

    return {
        "total_transactions": len(df),
        "total_alerts": len(alerts),
        "alert_rate_pct": round(len(alerts) / len(df) * 100, 2),
        "band_distribution": band_counts,
        "rule_fire_counts": rule_counts,
        "test_precision": round(precision, 4) if precision is not None else None,
        "test_recall": round(recall, 4) if recall is not None else None,
        "scored_csv_path": str(_SCORED_CSV),
    }


@router.post("/score", response_model=ScoreResponse)
def score_single(request: ScoreRequest) -> ScoreResponse:
    """Score a single raw bank transaction dict on-demand using the persisted model bundle.

    The transaction dict must contain the same columns as bank_anomaly.csv.
    CDR/IPDR context is not available for on-demand scoring — only bank features
    (Set A) are computed. The returned probability will be lower than the full
    Set C score for the same transaction.
    """
    if not _BUNDLE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model bundle not found at {_BUNDLE_PATH}. Run: python scripts/train.py",
        )
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))

    try:
        from scoring_core import (  # type: ignore[import]
            empty_cdr, empty_ipdr, load_bundle, score_frame,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Scoring module import failed: {e}")

    txn = request.transaction
    tid = txn.get("Transaction_ID", "UNKNOWN")

    try:
        bank_df = pd.DataFrame([txn])
        for col in ("Date", "Timestamp"):
            if col not in bank_df.columns:
                raise HTTPException(status_code=422, detail=f"Missing field: {col}")
        bank_df["ts"] = (
            pd.to_datetime(bank_df["Date"] + " " + bank_df["Timestamp"])
            .astype("int64") // 10**9
        )
        bank_df = bank_df.sort_values("ts").reset_index(drop=True)
        bank_df["y"] = 0

        # Same code path as the batch scorer, so a single transaction scored here
        # and the same transaction scored in a batch cannot disagree.
        scored = score_frame(bank_df, empty_cdr(), empty_ipdr(),
                             bundle=load_bundle(_BUNDLE_PATH), split="ondemand",
                             verbose=False)
        row = scored.iloc[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    return ScoreResponse(
        Transaction_ID=tid,
        ml_probability=round(float(row.ml_probability), 4),
        risk_score=float(row.risk_score),
        risk_band=str(row.risk_band),
        reasons=[r for r in (row.reason_1, row.reason_2, row.reason_3) if r],
        rules_fired=[r for r in str(row.rules_fired).split("|") if r],
    )
