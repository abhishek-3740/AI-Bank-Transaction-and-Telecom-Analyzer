"""Wire parsed PDFs into the investigation dashboard.

Before this module the parser was a dead end: /api/v1/pdf/parse returned rows to
the browser and deleted the upload, while every dashboard endpoint read CSVs that
only the offline scripts ever wrote. Uploading a statement could therefore never
change what the dashboard showed.

Responsibilities
----------------
1. Persist each parsed dataset under notebook/output/uploads/ so successive
   uploads accumulate (a bank statement and a CDR extract combine).
2. Normalise a parsed bank statement into the schema the feature builder needs —
   real statements omit timestamps, transaction IDs and beneficiary accounts,
   all of which the synthetic training data always carries.
3. Re-run scoring + graph analytics over everything uploaded so far and write the
   three CSVs the dashboard reads.

The uploaded corpus replaces the synthetic demo data in those CSVs. Regenerate
the demo baseline any time with: python scripts/score.py && python scripts/graph_analytics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

_BACKEND = Path(__file__).resolve().parents[1]      # backend/
_OUT = _BACKEND / "notebook" / "output"
_UPLOADS = _OUT / "uploads"
_SCRIPTS = _BACKEND / "scripts"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Canonical dataset key -> (uploaded csv name, date column, time column)
_DATASETS = {
    "BANK": ("bank_uploaded.csv", "Date", "Timestamp"),
    "CDR": ("cdr_uploaded.csv", "Call_Date", "Call_Start_Time"),
    "IPDR": ("ipdr_uploaded.csv", "Session_Date", "Session_Start_Time"),
}

# Statements rarely print a clock time against each line. Noon is deliberate: it
# is the neutral choice that keeps the ODD_HOUR rule (00:00-05:59) from firing on
# every single row of every statement, which would drown the alert queue.
# ponytail: time-of-day and short-window velocity features are meaningless for
# such rows. Upgrade path = parse the per-line time when the statement has one,
# and surface `timestamps_imputed` in the UI (already returned below).
_DEFAULT_TIME = "12:00:00"


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Infer which canonical dataset a parsed frame holds."""
    cols = set(df.columns)
    if {"Call_Date", "A_Party_Number"} & cols:
        return "CDR"
    if {"Session_Date", "Source_IP_Address"} & cols:
        return "IPDR"
    return "BANK"


def _coerce_time(series: pd.Series) -> tuple[pd.Series, int]:
    """Coerce a time column to strict HH:MM:SS, reporting how many were imputed."""
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    out = parsed.dt.strftime("%H:%M:%S")
    imputed = int(out.isna().sum())
    return out.fillna(_DEFAULT_TIME), imputed


def normalize_bank(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill the gaps a real bank statement leaves before feature building.

    The feature builder assumes the synthetic schema, where every transaction has
    an ID, a clock time, a sender customer ID and a beneficiary account. Parsed
    statements routinely have none of those. Returns the repaired frame plus a
    summary of what had to be inferred, so the caller can be honest about it.
    """
    df = df.copy()
    notes: dict = {}

    for col in ("Date", "Transaction_Amount"):
        if col not in df.columns:
            raise ValueError(f"Parsed statement has no '{col}' column; cannot score it.")

    # Drop statement furniture: opening/closing balance lines, carried-forward
    # rows and anything without both a date and an amount.
    df["Transaction_Amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    before = len(df)
    df = df[df["Date"].notna() & df["Transaction_Amount"].notna()].reset_index(drop=True)
    notes["rows_dropped_incomplete"] = before - len(df)
    if df.empty:
        raise ValueError("No transaction rows with both a date and an amount were found.")

    # The model was trained on unsigned amounts; statements sign them by DR/CR.
    # Keep the direction separately rather than discarding it.
    df["Direction"] = np.where(df["Transaction_Amount"] < 0, "DR", "CR")
    df["Transaction_Amount"] = df["Transaction_Amount"].abs()

    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.NA
    df["Timestamp"], imputed = _coerce_time(df["Timestamp"])
    notes["timestamps_imputed"] = imputed

    # A statement identifies one account holder; every row shares it.
    acct = _first_value(df, "Sender_Account_Number") or "UNKNOWN_ACCOUNT"
    for col, fallback in (("Sender_Customer_ID", acct),
                          ("Sender_Customer_Name", f"Account {str(acct)[-4:]}"),
                          ("Sender_Phone_Number", "")):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].fillna(_first_value(df, col) or fallback).astype(str)

    # Beneficiary accounts are usually absent; the narration is the only stable
    # handle on "same counterparty", which is what the graph and the
    # NEW_BENEFICIARY_FLAG rule actually need.
    if "Receiver_Account_Number" not in df.columns:
        df["Receiver_Account_Number"] = pd.NA
    narration = df.get("Receiver_Customer_Name")
    if narration is None:
        narration = pd.Series(pd.NA, index=df.index)
    narration = narration.fillna(df.get("Transaction_Mode", pd.Series("", index=df.index)))
    derived = "NARR:" + narration.fillna("UNKNOWN").astype(str).str.strip().str.upper()
    notes["receivers_derived_from_narration"] = int(df["Receiver_Account_Number"].isna().sum())
    df["Receiver_Account_Number"] = df["Receiver_Account_Number"].fillna(derived).astype(str)

    # Deterministic IDs so re-uploading the same statement does not duplicate rows.
    if "Transaction_ID" not in df.columns:
        df["Transaction_ID"] = pd.NA
    generated = (str(acct) + "-" + df["Date"].str.replace("-", "", regex=False)
                 + "-" + df.index.astype(str).str.zfill(4))
    notes["transaction_ids_generated"] = int(df["Transaction_ID"].isna().sum())
    df["Transaction_ID"] = df["Transaction_ID"].fillna(generated).astype(str)

    return df, notes


def _first_value(df: pd.DataFrame, col: str):
    """First non-null value of a column, or None when absent/empty."""
    if col not in df.columns:
        return None
    non_null = df[col].dropna()
    return non_null.iloc[0] if len(non_null) else None


def _add_ts(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
    """Attach the epoch-second `ts` column every feature lookup indexes on."""
    df = df.copy()
    ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                        errors="coerce")
    df = df[ts.notna()].reset_index(drop=True)
    df["ts"] = ts.dropna().astype("int64").to_numpy() // 10**9
    return df.sort_values("ts").reset_index(drop=True)


def persist_upload(df: pd.DataFrame, dataset_type: str) -> int:
    """Merge a parsed frame into the accumulated upload corpus. Returns total rows."""
    filename, _, _ = _DATASETS[dataset_type]
    path = _UPLOADS / filename
    _UPLOADS.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        df = pd.concat([existing, df.astype(str)], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Upload corpus {filename} now holds {len(df)} rows")
    return len(df)


def _load_corpus(dataset_type: str) -> pd.DataFrame | None:
    filename, date_col, time_col = _DATASETS[dataset_type]
    path = _UPLOADS / filename
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return None
    if dataset_type == "BANK":
        df, _ = normalize_bank(df)
    else:
        if time_col not in df.columns:
            df[time_col] = _DEFAULT_TIME
        df[time_col], _ = _coerce_time(df[time_col])
    return _add_ts(df, date_col, time_col)


def rebuild_dashboard() -> dict:
    """Re-score every uploaded transaction and rewrite the dashboard CSVs."""
    from scoring_core import empty_cdr, empty_ipdr, load_bundle, score_frame
    from graph_analytics import build_financial_graph

    bank = _load_corpus("BANK")
    if bank is None:
        return {"dashboard_updated": False,
                "reason": "No bank statement uploaded yet — the dashboard scores "
                          "bank transactions, so upload a bank statement PDF."}

    cdr = _load_corpus("CDR")
    ipdr = _load_corpus("IPDR")
    bank = bank.copy()
    bank["y"] = 0          # uploaded evidence carries no ground-truth labels

    scored = score_frame(bank, cdr if cdr is not None else empty_cdr(),
                         ipdr if ipdr is not None else empty_ipdr(),
                         bundle=load_bundle(), split="upload", verbose=False)

    _OUT.mkdir(parents=True, exist_ok=True)
    scored.to_csv(_OUT / "scored_transactions.csv", index=False)

    # build_financial_graph expects a ground-truth frame; uploads have none, so
    # an empty one leaves is_mule_account at 0 rather than inventing labels.
    no_gt = pd.DataFrame({"Transaction_ID": pd.Series(dtype=str)})
    nodes, edges = build_financial_graph(bank, scored, no_gt)
    nodes.to_csv(_OUT / "graph_analytics.csv", index=False)
    edges.to_csv(_OUT / "graph_edges.csv", index=False)
    # No cache invalidation needed: the routers key their caches on file mtime.

    return {
        "dashboard_updated": True,
        "scored_transactions": len(scored),
        "alerts": int((scored["risk_score"] >= 70).sum()),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "cdr_context": cdr is not None,
        "ipdr_context": ipdr is not None,
    }


def ingest(df: pd.DataFrame, dataset_type: str) -> dict:
    """Persist a parsed frame and refresh the dashboard from the whole corpus."""
    summary: dict = {"dataset_type": dataset_type}
    working = df

    if dataset_type == "BANK":
        # Normalise before persisting so the stored corpus is already scoreable
        # and the caller learns what had to be inferred from this upload.
        working, notes = normalize_bank(df)
        summary.update(notes)

    summary["corpus_rows"] = persist_upload(working, dataset_type)
    summary.update(rebuild_dashboard())
    return summary
