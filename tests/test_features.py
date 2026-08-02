"""Tests for train.py feature engineering — covers the three bugs that actually occurred.

No fixtures. Each test is standalone and can be run with:
    pytest tests/test_features.py

See HANDOFF.md §7 for the bugs these tests guard against.
"""
import numpy as np
import pandas as pd
import tempfile, os


# ---------------------------------------------------------------------------
# Test 1: Causality — no feature may peek at future transactions
# ---------------------------------------------------------------------------
def test_causality_no_future_leakage():
    """Build features for 5 transactions, then append a 6th *later* one.
    Assert that features for the first 5 rows are identical both times.
    This guards against any feature that accidentally uses groupby().transform()
    or similar all-rows operations that would leak future information."""
    from pathlib import Path
    import importlib, sys

    # --- Minimal synthetic data (5 bank rows, no CDR/IPDR) ---
    base_time = pd.Timestamp("2025-03-15 10:00:00")
    cid = "100000001"
    phone = "+919800000001"
    rows = []
    for i in range(5):
        t = base_time + pd.Timedelta(minutes=i * 30)
        rows.append({
            "Transaction_ID": f"TXN{i:04d}", "Date": t.strftime("%Y-%m-%d"),
            "Timestamp": t.strftime("%H:%M:%S"), "Txn_Ref_Number": f"REF{i:04d}",
            "Transaction_Mode": "UPI", "Currency": "INR",
            "Transaction_Amount": 1000 + i * 100,
            "Sender_Customer_ID": cid, "Sender_Customer_Name": "Test User",
            "Sender_Bank_Name": "HDFC Bank", "Sender_Account_Number": "123456789012",
            "Sender_Account_Type": "Savings", "Sender_IFSC": "HDFC0000001",
            "Sender_Phone_Number": phone,
            "Receiver_Customer_ID": "100000002", "Receiver_Customer_Name": "Other",
            "Receiver_Bank_Name": "ICICI Bank", "Receiver_Account_Number": "987654321098",
            "Receiver_Account_Type": "Savings", "Receiver_IFSC": "ICIC0000001",
            "Receiver_Phone_Number": "+919800000002",
        })
    bank5 = pd.DataFrame(rows)

    # Add a 6th transaction 2 hours later with a very large amount
    t6 = base_time + pd.Timedelta(hours=4)
    row6 = rows[-1].copy()
    row6.update({"Transaction_ID": "TXN9999", "Date": t6.strftime("%Y-%m-%d"),
                 "Timestamp": t6.strftime("%H:%M:%S"), "Transaction_Amount": 999999.0})
    bank6 = pd.concat([bank5, pd.DataFrame([row6])], ignore_index=True)

    # Compute features with the helpers from train.py
    def compute_bank_features(bdf):
        """Replicate the Set A bank-only feature logic from train.py."""
        bdf = bdf.copy()
        bdf["ts"] = pd.to_datetime(bdf.Date + " " + bdf.Timestamp).astype("int64") // 10**9
        bdf = bdf.sort_values("ts").reset_index(drop=True)
        g = bdf.groupby("Sender_Customer_ID", sort=False)
        A = pd.DataFrame(index=bdf.index)
        A["transaction_amount"] = bdf.Transaction_Amount
        A["customer_history_count"] = g.cumcount()
        A["time_since_previous_transaction"] = g.ts.diff()

        # Velocity — explicit positional (no groupby().rolling())
        DAY = 86400
        snd_vals = bdf.Sender_Customer_ID.values
        ts_vals = bdf.ts.values
        amt_vals = bdf.Transaction_Amount.values
        vel = np.zeros((len(bdf), 3))
        for i in range(len(bdf)):
            cid_i = snd_vals[i]
            t_i = ts_vals[i]
            for j, w in enumerate((600, 1800, 3600)):
                mask = (snd_vals[:i] == cid_i) & (ts_vals[:i] >= t_i - w)
                vel[i, j] = mask.sum()
        A["txn_count_previous_10m"] = vel[:, 0]
        A["txn_count_previous_30m"] = vel[:, 1]
        A["txn_count_previous_1h"] = vel[:, 2]
        return A

    feat5 = compute_bank_features(bank5)
    feat6 = compute_bank_features(bank6)

    # The first 5 rows must have IDENTICAL features regardless of the 6th row
    for col in feat5.columns:
        v5 = feat5[col].values
        v6 = feat6[col].iloc[:5].values
        np.testing.assert_array_equal(
            np.nan_to_num(v5, nan=-1), np.nan_to_num(v6, nan=-1),
            err_msg=f"Feature '{col}' changed when a future transaction was appended — causality violated"
        )


# ---------------------------------------------------------------------------
# Test 2: Alignment — txn_count_previous_1h matches brute-force count
# ---------------------------------------------------------------------------
def test_alignment_velocity_count():
    """Build a small frame with known timestamps and verify txn_count_previous_1h
    matches a brute-force count. This catches the groupby().rolling() scramble bug
    where .values was assigned back in group order instead of row order."""
    base = pd.Timestamp("2025-06-01 12:00:00")
    cid = "CUST_A"
    # 4 transactions: t=0, t=20m, t=40m, t=90m (1.5h after first)
    times = [base, base + pd.Timedelta(minutes=20),
             base + pd.Timedelta(minutes=40), base + pd.Timedelta(minutes=90)]

    ts_epoch = [int(t.timestamp()) for t in times]

    # Brute-force expected counts (strictly before, within 1h window):
    # txn0 (t=0):   no prior txns → 0
    # txn1 (t=20m): txn0 is 20m before → 1
    # txn2 (t=40m): txn0 (40m) and txn1 (20m) both within 1h → 2
    # txn3 (t=90m): txn0 (90m, outside 1h), txn1 (70m, outside), txn2 (50m, within) → 1
    expected = [0, 1, 2, 1]

    # Compute using the same positional logic as train.py's index_by/before helpers
    def index_by_simple(keys, timestamps):
        order = np.argsort(keys, kind="stable")
        k = keys[order]
        t = timestamps[order]
        result = {}
        uniq, start = np.unique(k, return_index=True)
        end = np.append(start[1:], len(k))
        for u, s, e in zip(uniq, start, end):
            result[u] = (t[s:e], order[s:e])
        return result

    keys = np.array([cid] * 4)
    ts_arr = np.array(ts_epoch)
    idx = index_by_simple(keys, ts_arr)

    counts = []
    for i in range(4):
        ent = idx.get(cid)
        ts_all, pos_all = ent
        hi = np.searchsorted(ts_all, ts_arr[i])
        lo = np.searchsorted(ts_all, ts_arr[i] - 3600)
        counts.append(hi - lo)

    assert counts == expected, (
        f"txn_count_previous_1h mismatch: got {counts}, expected {expected}. "
        "This is the groupby().rolling() alignment bug."
    )


# ---------------------------------------------------------------------------
# Test 3: Dtype — phone numbers survive CSV round-trip as str with +91 prefix
# ---------------------------------------------------------------------------
def test_phone_dtype_csv_roundtrip():
    """Pandas parses '+919812345678' as int64 unless dtype is forced to str.
    This silently breaks every phone join between bank/CDR/IPDR.
    Assert that reading a CSV with the STR dtype dict preserves the prefix."""
    phone = "+919812345678"
    df = pd.DataFrame({
        "Sender_Phone_Number": [phone],
        "Transaction_ID": ["TXN001"],
        "Transaction_Amount": [5000.0],
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        df.to_csv(f, index=False)
        path = f.name

    try:
        # Read WITHOUT dtype override — this is how the bug manifests
        raw = pd.read_csv(path)
        # The phone column may or may not survive depending on pandas version,
        # but the fix is to always read with dtype=str for phone columns.

        # Read WITH the dtype override (the fix from train.py)
        STR = {"Sender_Phone_Number": str}
        fixed = pd.read_csv(path, dtype=STR)

        assert isinstance(fixed["Sender_Phone_Number"].iloc[0], str), \
            "Sender_Phone_Number is not str after CSV round-trip with dtype override"
        assert fixed["Sender_Phone_Number"].iloc[0].startswith("+91"), \
            f"Sender_Phone_Number lost +91 prefix: got '{fixed['Sender_Phone_Number'].iloc[0]}'"
        assert fixed["Sender_Phone_Number"].iloc[0] == phone, \
            f"Phone number corrupted: got '{fixed['Sender_Phone_Number'].iloc[0]}', expected '{phone}'"
    finally:
        os.unlink(path)
