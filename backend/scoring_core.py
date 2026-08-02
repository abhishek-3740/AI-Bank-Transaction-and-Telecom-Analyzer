"""Thin scoring wrapper used by pdf/ingest.py to re-score uploaded data.

Loads the persisted model bundle once (module-level cache) and exposes the
functions rebuild_dashboard() needs without reimplementing feature engineering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND = Path(__file__).resolve().parent          # backend/
_SCRIPTS = _BACKEND / "scripts"
_MODELS  = _BACKEND / "models"
_BUNDLE_PATH = _MODELS / "stage7_setC.joblib"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# ── module-level cache so we only deserialise the joblib once per process ──
_bundle_cache: dict | None = None


def load_bundle() -> dict:
    """Load (and cache) the persisted model bundle."""
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache
    if not _BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {_BUNDLE_PATH}. "
            "Run: python scripts/train.py"
        )
    import joblib
    _bundle_cache = joblib.load(_BUNDLE_PATH)
    return _bundle_cache


def empty_cdr() -> pd.DataFrame:
    """Return an empty CDR frame with the required columns."""
    import features as _f  # noqa: F401 — triggers the import so _f.index_by works
    cols = [
        "CDR_ID", "Call_Date", "Call_Start_Time",
        "A_Party_Number", "B_Party_Number", "Call_Type",
        "Call_Duration_Seconds", "IMSI", "IMEI",
        "First_BTS_Location", "First_Cell_Global_ID",
        "Roaming_Network_Circle", "ts",
    ]
    return pd.DataFrame(columns=cols)


def empty_ipdr() -> pd.DataFrame:
    """Return an empty IPDR frame with the required columns."""
    cols = [
        "IPDR_ID", "Session_Date", "Session_Start_Time",
        "Subscriber_IMSI", "Subscriber_MSISDN", "Device_IMEI",
        "Source_IP_Address", "Destination_IP_Address",
        "Destination_Port", "Cell_Global_ID",
        "Session_Duration_Seconds", "ts",
    ]
    return pd.DataFrame(columns=cols)


def score_frame(
    bank: pd.DataFrame,
    cdr: pd.DataFrame,
    ipdr: pd.DataFrame,
    bundle: dict,
    split: str = "upload",
    verbose: bool = False,
) -> pd.DataFrame:
    """Score a bank frame (+ optional CDR/IPDR context) and return the full
    scored_transactions.csv-compatible DataFrame.

    Parameters
    ----------
    bank : pd.DataFrame
        Must already carry a `ts` epoch-second column and `y` label column.
    cdr, ipdr : pd.DataFrame
        Contextual records. Pass empty_cdr()/empty_ipdr() when absent.
    bundle : dict
        The joblib bundle written by scripts/train.py.
    split : str
        Label to stamp in the `split` column (e.g. "upload").
    verbose : bool
        Forward to build_features().
    """
    import xgboost as xgb
    from features import build_features

    # Ensure join-key columns are str so phone matching works
    STR_COLS = [
        "Sender_Phone_Number", "Receiver_Phone_Number",
        "Sender_Account_Number", "Receiver_Account_Number",
        "Sender_Customer_ID", "Receiver_Customer_ID",
        "A_Party_Number", "B_Party_Number",
        "IMSI", "IMEI", "Subscriber_IMSI",
        "Subscriber_MSISDN", "Device_IMEI",
    ]
    for df in (bank, cdr, ipdr):
        for col in STR_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # Add ts to CDR / IPDR if they are non-empty and missing it
    for df, dcol, tcol in (
        (cdr,  "Call_Date",     "Call_Start_Time"),
        (ipdr, "Session_Date",  "Session_Start_Time"),
    ):
        if not df.empty and "ts" not in df.columns:
            ts = pd.to_datetime(
                df[dcol].astype(str) + " " + df[tcol].astype(str),
                errors="coerce",
            )
            df["ts"] = ts.astype("int64") // 10**9

    b, SETS = build_features(bank, cdr, ipdr, verbose=verbose)

    X = SETS["C"].replace([np.inf, -np.inf], np.nan)[bundle["columns"]]
    X = X.fillna(bundle["medians"]).fillna(0)

    model, thr = bundle["model"], bundle["threshold"]
    prob = model.predict_proba(X)[:, 1]

    # 0-100 risk score, piecewise-linear, threshold pinned to 70
    risk = np.where(
        prob < thr,
        70 * prob / thr,
        70 + 30 * (prob - thr) / (1 - thr + 1e-12),
    )
    risk = np.clip(risk, 0, 100).round(1)
    band = pd.cut(
        risk, [-0.1, 40, 70, 90, 100],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )

    # TreeSHAP top-3 reasons
    contrib = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=list(X.columns)),
        pred_contribs=True,
    )[:, :-1]
    top3 = np.argsort(-contrib, axis=1)[:, :3]
    cols_arr = np.array(X.columns)
    reasons = [
        [
            f"{cols_arr[j]} ({contrib[i, j]:+.2f})" if contrib[i, j] > 0 else ""
            for j in top3[i]
        ]
        for i in range(len(X))
    ]

    # Rule engine
    hour       = X["transaction_hour"].values
    amt_ratio  = X["amount_vs_customer_median"].fillna(0).values
    rapid      = X["txn_count_previous_10m"].values
    new_ben    = X["receiver_seen_before"].values
    calls_30m  = X["calls_previous_30m"].values
    rule_parts = [
        ("ODD_HOUR",             (hour >= 0) & (hour <= 5)),
        ("HIGH_AMOUNT_ANOMALY",  amt_ratio > 5.0),
        ("RAPID_SUCCESSION",     rapid >= 3),
        ("NEW_BENEFICIARY_FLAG", new_ben == 0),
        ("TELECOM_BURST",        calls_30m >= 3),
    ]
    rules_fired = []
    for masks in zip(*[m for _, m in rule_parts]):
        fired = [name for (name, _), hit in zip(rule_parts, masks) if hit]
        rules_fired.append("|".join(fired))

    scored = pd.DataFrame({
        "Transaction_ID":        b.Transaction_ID,
        "Date":                  b.Date,
        "Timestamp":             b.Timestamp,
        "Sender_Customer_ID":    b.Sender_Customer_ID,
        "Sender_Customer_Name":  b.Sender_Customer_Name,
        "Receiver_Account_Number": b.Receiver_Account_Number,
        "Transaction_Amount":    b.Transaction_Amount,
        "ml_probability":        prob.round(4),
        "risk_score":            risk,
        "risk_band":             band,
        "reason_1":              [r[0] for r in reasons],
        "reason_2":              [r[1] for r in reasons],
        "reason_3":              [r[2] for r in reasons],
        "rules_fired":           rules_fired,
        "split":                 split,
        "is_suspicious_gt":      b.y,
    }).sort_values("risk_score", ascending=False).reset_index(drop=True)

    return scored
