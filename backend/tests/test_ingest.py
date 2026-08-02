"""Tests for the PDF -> dashboard ingestion path.

Guards the bug this module was written to fix: parsing a PDF used to leave the
dashboard untouched, because nothing persisted the parsed rows or re-scored them.

No fixtures. Run with:
    pytest tests/test_ingest.py
"""
import sys
from pathlib import Path

import pandas as pd

_BACKEND = Path(__file__).resolve().parents[1]
for p in (str(_BACKEND), str(_BACKEND / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _statement_as_parsed() -> pd.DataFrame:
    """A bank statement shaped like real parser output: no timestamps, no
    transaction IDs, no beneficiary accounts, and summary rows still present."""
    return pd.DataFrame({
        "Transaction_ID": [None, None, None, None],
        "Date": [None, "2024-07-30", "2024-08-03", "2024-08-04"],
        "Timestamp": [None, None, None, None],
        "Transaction_Mode": ["OPENING_BALANCE", "UPI", "ATM", "UPI"],
        "Transaction_Amount": [None, "51000.0", "-10000.0", "295.0"],
        "Sender_Customer_ID": ["966596607"] * 4,
        "Sender_Account_Number": ["924020031969007"] * 4,
        "Sender_Bank_Name": ["Axis Bank"] * 4,
        "Receiver_Account_Number": [None, None, None, None],
    })


def test_normalize_bank_fills_what_statements_omit():
    """Real statements omit the fields the feature builder assumes exist.
    Normalisation must supply all of them, drop rows that carry no transaction,
    and report honestly what it had to infer."""
    from pdf.ingest import normalize_bank

    norm, notes = normalize_bank(_statement_as_parsed())

    # The opening-balance row has no date and no amount — it is not a transaction.
    assert len(norm) == 3, f"expected 3 transaction rows, got {len(norm)}"
    assert notes["rows_dropped_incomplete"] == 1

    # Every column build_features indexes on must now be present and complete.
    for col in ("Transaction_ID", "Date", "Timestamp", "Sender_Customer_ID",
                "Sender_Customer_Name", "Sender_Phone_Number",
                "Receiver_Account_Number", "Transaction_Amount"):
        assert col in norm.columns, f"{col} missing after normalisation"
        assert norm[col].notna().all(), f"{col} still has nulls after normalisation"

    # Timestamps must be strict HH:MM:SS — build_features parses with that exact
    # format and raises on anything else.
    assert (norm["Timestamp"].str.match(r"^\d{2}:\d{2}:\d{2}$")).all()
    assert notes["timestamps_imputed"] == 3

    # The imputed hour must not trip the ODD_HOUR rule (00:00-05:59), or every
    # row of every statement would alert.
    assert not norm["Timestamp"].str.startswith(("00", "01", "02", "03", "04", "05")).any()

    # Transaction IDs must be unique, or velocity and graph aggregation collapse.
    assert norm["Transaction_ID"].is_unique

    # The model was trained on unsigned amounts; direction is kept, not discarded.
    assert (norm["Transaction_Amount"] >= 0).all()
    assert norm["Direction"].tolist() == ["CR", "DR", "CR"]


def test_normalize_bank_rejects_a_statement_with_no_transactions():
    """A statement whose rows all lack a date or an amount cannot be scored;
    failing loudly beats writing an empty dashboard."""
    from pdf.ingest import normalize_bank

    empty = _statement_as_parsed().iloc[[0]]          # opening balance only
    try:
        normalize_bank(empty)
    except ValueError as exc:
        assert "no transaction rows" in str(exc).lower() or "found" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for a statement with no transactions")


def test_dataset_type_detection():
    from pdf.ingest import detect_dataset_type

    assert detect_dataset_type(_statement_as_parsed()) == "BANK"
    assert detect_dataset_type(pd.DataFrame(columns=["Call_Date", "A_Party_Number"])) == "CDR"
    assert detect_dataset_type(
        pd.DataFrame(columns=["Session_Date", "Source_IP_Address"])) == "IPDR"


def test_normalized_statement_survives_feature_building_and_scoring():
    """The whole point: a normalised statement must actually reach a risk score.
    This is what silently never happened before — the parse result went nowhere."""
    from pdf.ingest import _add_ts, normalize_bank
    from scoring_core import empty_cdr, empty_ipdr, load_bundle, score_frame

    bundle_path = _BACKEND / "models" / "stage7_setC.joblib"
    if not bundle_path.exists():
        return  # model bundle not trained in this checkout; nothing to assert

    norm, _ = normalize_bank(_statement_as_parsed())
    bank = _add_ts(norm, "Date", "Timestamp")
    bank["y"] = 0

    scored = score_frame(bank, empty_cdr(), empty_ipdr(),
                         bundle=load_bundle(bundle_path), split="upload", verbose=False)

    assert len(scored) == len(bank)
    assert scored["risk_score"].between(0, 100).all()
    assert set(scored["risk_band"]) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    # These columns are exactly what the dashboard endpoints read back.
    for col in ("Transaction_ID", "risk_score", "risk_band", "rules_fired",
                "split", "is_suspicious_gt"):
        assert col in scored.columns


if __name__ == "__main__":
    test_normalize_bank_fills_what_statements_omit()
    test_normalize_bank_rejects_a_statement_with_no_transactions()
    test_dataset_type_detection()
    test_normalized_statement_survives_feature_building_and_scoring()
    print("all ingest checks passed")
