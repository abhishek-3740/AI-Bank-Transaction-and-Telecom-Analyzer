#!/usr/bin/env python
"""Shared scoring implementation for TRI-NETRA.

Imported by BOTH scripts/score.py (offline batch over the synthetic dataset) and
pdf/ingest.py (online scoring of an uploaded statement), for the same reason
features.py is shared: a second copy of the risk mapping / SHAP reasons / rule
engine is how the scores silently drift apart (HANDOFF.md section 3.2).

Never retrains. Reuses the bundle's imputation medians, because refitting them
here would leak and shift every score.
"""
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

from features import build_features

ROOT = Path(__file__).resolve().parents[1]   # backend/
BUNDLE_PATH = ROOT / "models" / "stage7_setC.joblib"

# Rules fire on raw transaction/feature signals the ML model cannot rank
# reliably on its own (HANDOFF.md section 3.1). Each entry maps a rule name to a
# predicate over the feature matrix.
RULES = [
    ("ODD_HOUR",             lambda X: (X["transaction_hour"] >= 0) & (X["transaction_hour"] <= 5)),
    ("HIGH_AMOUNT_ANOMALY",  lambda X: X["amount_vs_customer_median"].fillna(0) > 5.0),
    ("RAPID_SUCCESSION",     lambda X: X["txn_count_previous_10m"] >= 3),
    ("NEW_BENEFICIARY_FLAG", lambda X: X["receiver_seen_before"] == 0),
    ("TELECOM_BURST",        lambda X: X["calls_previous_30m"] >= 3),
]


def empty_cdr() -> pd.DataFrame:
    """A CDR frame with the columns build_features touches but no rows.

    Used whenever bank data arrives without telecom context (single-transaction
    scoring, uploaded statements): Set B/C features resolve to "no context"
    instead of the feature builder raising on a missing column.
    """
    df = pd.DataFrame(columns=["CDR_ID", "Call_Date", "Call_Start_Time",
                               "A_Party_Number", "B_Party_Number", "Call_Type",
                               "Call_Duration_Seconds", "IMSI", "IMEI",
                               "First_Cell_Global_ID", "Roaming_Network_Circle"],
                      dtype=object)
    df["ts"] = pd.Series(dtype="int64")
    return df


def empty_ipdr() -> pd.DataFrame:
    """An IPDR frame with the columns build_features touches but no rows."""
    df = pd.DataFrame(columns=["IPDR_ID", "Session_Date", "Session_Start_Time",
                               "Subscriber_MSISDN", "Subscriber_IMSI", "Device_IMEI",
                               "Source_IP_Address", "Destination_IP_Address",
                               "Destination_Port", "Cell_Global_ID",
                               "Session_Duration_Seconds"], dtype=object)
    df["ts"] = pd.Series(dtype="int64")
    return df


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    """Load the persisted Stage 7 model bundle, or raise with a fixable message."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run: python scripts/train.py")
    bundle = joblib.load(path)
    if bundle["kind"] != "XGBoost":
        raise NotImplementedError(
            f"bundle holds a '{bundle['kind']}' model; only the plain XGBoost path is "
            "implemented here. Extend scoring_core.py before switching the selected model.")
    return bundle


def prepare_matrix(X: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Align a feature frame to the bundle's training columns and impute."""
    X = X.replace([np.inf, -np.inf], np.nan)
    # Uploaded statements carry no CDR/IPDR context, so those columns can be
    # absent entirely; the bundle medians stand in for them.
    for col in bundle["columns"]:
        if col not in X.columns:
            X[col] = np.nan
    return X[bundle["columns"]].fillna(bundle["medians"]).fillna(0)


def risk_from_probability(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Piecewise-linear map pinning the tuned threshold to 70, so "risk >= 70"
    means "the model would alert" rather than an arbitrary cutoff."""
    risk = np.where(prob < threshold,
                    70 * prob / threshold,
                    70 + 30 * (prob - threshold) / (1 - threshold + 1e-12))
    return np.clip(risk, 0, 100).round(1)


def band_from_risk(risk: np.ndarray) -> pd.Categorical:
    return pd.cut(risk, [-0.1, 40, 70, 90, 100],
                  labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])


def top_reasons(model, X: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """Exact per-transaction TreeSHAP contributions; last column is the bias term."""
    contrib = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)[:, :-1]
    top = np.argsort(-contrib, axis=1)[:, :k]
    cols = np.array(X.columns)
    return [[f"{cols[j]} ({contrib[i, j]:+.2f})" if contrib[i, j] > 0 else ""
             for j in top[i]] for i in range(len(X))]


def fire_rules(X: pd.DataFrame) -> np.ndarray:
    """Return a '|'-joined rule-name string per row."""
    masks = {}
    for name, predicate in RULES:
        try:
            masks[name] = np.asarray(predicate(X)).astype(bool)
        except KeyError:
            # Feature unavailable for this dataset (e.g. no CDR context) — the
            # rule simply cannot fire rather than blocking the whole scoring run.
            masks[name] = np.zeros(len(X), dtype=bool)
    return np.array(["|".join(n for n in masks if masks[n][i]) for i in range(len(X))])


def temporal_split(ts: pd.Series) -> np.ndarray:
    """Same temporal boundaries as training, so the report can mark which rows
    the model actually held out. Scores on train-period rows are in-sample."""
    cut_tr, cut_va = np.quantile(ts, 0.60), np.quantile(ts, 0.75)
    return np.where(ts < cut_tr, "train", np.where(ts < cut_va, "val", "test"))


def score_frame(bank: pd.DataFrame, cdr: pd.DataFrame, ipdr: pd.DataFrame,
                bundle: dict | None = None, split: str | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """Score a bank frame and return the investigator-facing scored table.

    Args:
        bank/cdr/ipdr: source frames carrying an epoch-second ``ts`` column.
        bundle: preloaded model bundle; loaded from disk when omitted.
        split: constant split label (e.g. "upload"). When None, the training
            temporal split is recomputed — only meaningful for the dataset the
            model was actually trained on.
    """
    bundle = bundle or load_bundle()
    b, SETS = build_features(bank, cdr, ipdr, verbose=verbose)
    X = prepare_matrix(SETS["C"], bundle)

    prob = bundle["model"].predict_proba(X)[:, 1]
    risk = risk_from_probability(prob, bundle["threshold"])
    reasons = top_reasons(bundle["model"], X)

    return pd.DataFrame({
        "Transaction_ID": b.Transaction_ID, "Date": b.Date, "Timestamp": b.Timestamp,
        "Sender_Customer_ID": b.Sender_Customer_ID,
        "Sender_Customer_Name": b.Sender_Customer_Name,
        "Receiver_Account_Number": b.Receiver_Account_Number,
        "Transaction_Amount": b.Transaction_Amount,
        "ml_probability": prob.round(4), "risk_score": risk,
        "risk_band": band_from_risk(risk),
        "reason_1": [r[0] for r in reasons], "reason_2": [r[1] for r in reasons],
        "reason_3": [r[2] for r in reasons],
        "rules_fired": fire_rules(X),
        "split": split if split is not None else temporal_split(b.ts),
        "is_suspicious_gt": b.y if "y" in b else 0,
    }).sort_values("risk_score", ascending=False).reset_index(drop=True)
