#!/usr/bin/env python
"""Regenerate the TRI-NETRA datasets.

The original baseline had no structure for anomalies to deviate from: transaction
hours were uniform across 24h, 99.8% of transactions went to a never-seen
beneficiary, and amounts had no per-customer profile. Injected anomalies
therefore sat *inside* the baseline noise -- every canonical fraud feature
scored AUC 0.46-0.60, and 57 of the 100 labelled rows were byte-identical to the
clean baseline.

This rebuilds the baseline with that structure, then injects deviations
calibrated to clear it. CSV schema is unchanged; only the statistics are.
Run: python scripts/generate_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

N_CUSTOMERS = 400
DAYS = 90
START = datetime(2025, 1, 1)
N_BANK = 20_000
N_CDR = 60_000
N_IPDR = 50_000
N_ANOMALY = 800

ROOT = Path(__file__).resolve().parents[1]   # backend/
OUT  = ROOT.parent / "data"                  # data/ is at repo root

# Daytime-weighted transaction/call clock. Hours 1-4 hold ~1.9% of activity, so
# an odd-hour injection is a real outlier (it was 24.9% before).
HOUR_W = np.array([.008, .005, .004, .004, .006, .014, .030, .048, .065, .080, .088, .084,
                   .078, .072, .070, .066, .062, .058, .052, .045, .036, .026, .017, .010])
HOUR_W = HOUR_W / HOUR_W.sum()

BANKS = [("HDFC Bank", "HDFC"), ("ICICI Bank", "ICIC"), ("State Bank of India", "SBIN"),
         ("Axis Bank", "UTIB"), ("Punjab National Bank", "PUNB"), ("Canara Bank", "CNRB"),
         ("IndusInd Bank", "INDB"), ("Bank of Baroda", "BARB"),
         ("Kotak Mahindra Bank", "KKBK"), ("Union Bank of India", "UBIN")]
ACCT_TYPES, ACCT_W = ["Savings", "Salary", "Current", "Demat"], [.64, .27, .055, .035]
MODES = ["ATM", "UPI", "Cash Deposit", "IMPS", "Pos", "Cash", "NEFT", "RTGS"]
MODE_W = [.352, .337, .218, .085, .007, .0008, .0004, .0002]
MODE_W = np.array(MODE_W) / np.sum(MODE_W)
CIRCLES = ["West Bengal", "Rajasthan", "Delhi", "Mumbai", "Gujarat", "Maharashtra",
           "Uttar Pradesh", "Karnataka", "Tamil Nadu"]
BTS_PER_CIRCLE = 10
PORTS, PORT_W = [443, 80, 53, 123, 8080, 8443, 5228], [.50, .20, .10, .05, .05, .05, .05]
FIRST = ["Rohan", "Niraj", "Sarthak", "Kaushik", "Aditya", "Ishaan", "Meera", "Ananya",
         "Vikram", "Priya", "Arjun", "Kavya", "Rahul", "Sneha", "Manish", "Divya"]
LAST = ["Pandey", "Rathod", "Agarwal", "Varghese", "Sharma", "Iyer", "Nair", "Reddy",
        "Bose", "Chauhan", "Malhotra", "Joshi", "Menon", "Kulkarni", "Banerjee", "Gupta"]
ALNUM = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))


def codes(n, k):
    """n random k-char alphanumeric strings."""
    return np.ascontiguousarray(rng.choice(ALNUM, size=(n, k))).view(f"<U{k}").ravel()


def digits(n, k, first=None):
    """n random k-digit strings, optionally with a fixed first digit set."""
    d = rng.integers(0, 10, size=(n, k)).astype("<U1")
    if first is not None:
        d[:, 0] = rng.choice(list(first), n)
    return np.ascontiguousarray(d).view(f"<U{k}").ravel()


def ragged(sizes, pool):
    """Per-entity sets of distinct members drawn from `pool`, in flat form."""
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    flat = np.concatenate([rng.choice(pool, s, replace=False) for s in sizes])
    return flat, offsets


def pick(flat, offsets, sizes, idx):
    """Draw one member from each entity's set (vectorised over idx)."""
    return flat[offsets[idx] + (rng.random(len(idx)) * sizes[idx]).astype(int)]


def stamps(n, owner=None, days=DAYS, base=START):
    """Random datetimes. With `owner`, hours come from that customer's own clock
    so "unusual hour" is defined per customer, not just globally."""
    if owner is None:
        hour = rng.choice(24, n, p=HOUR_W)
    else:
        hour = (rng.random(n)[:, None] > HOUR_CW[owner]).sum(1).clip(0, 23)
    sec = rng.integers(0, days, n) * 86400 + hour * 3600 + rng.integers(0, 3600, n)
    return np.array([base + timedelta(seconds=int(s)) for s in sec])


def dcol(dt):
    return pd.Series(dt).dt.strftime("%Y-%m-%d").values


def tcol(dt):
    return pd.Series(dt).dt.strftime("%H:%M:%S").values


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------
cust = pd.DataFrame({
    "cid": (100000000 + np.arange(N_CUSTOMERS)).astype(str),
    "name": [f"{a} {b}" for a, b in zip(rng.choice(FIRST, N_CUSTOMERS), rng.choice(LAST, N_CUSTOMERS))],
    "acct": digits(N_CUSTOMERS, 12, first="123456789"),
    "phone": np.char.add("+91", digits(N_CUSTOMERS, 10, first="6789")),
    "acct_type": rng.choice(ACCT_TYPES, N_CUSTOMERS, p=ACCT_W),
    "imsi": np.char.add("404", digits(N_CUSTOMERS, 12)),
    "imei": digits(N_CUSTOMERS, 15, first="35"),
    "circle": rng.choice(len(CIRCLES), N_CUSTOMERS),
    "bts": rng.integers(0, BTS_PER_CIRCLE, N_CUSTOMERS),
    # per-customer log-normal amount profile: this is what makes a "relative
    # spike" meaningful. Absent before -- all customers shared one distribution.
    "mu": rng.normal(9.4, 0.45, N_CUSTOMERS),
    "sigma": rng.uniform(0.50, 0.85, N_CUSTOMERS),
})
bank_pick = rng.choice(len(BANKS), N_CUSTOMERS)
cust["bank"] = [BANKS[i][0] for i in bank_pick]
cust["ifsc"] = np.char.add([BANKS[i][1] for i in bank_pick], digits(N_CUSTOMERS, 7))
cust["bts_name"] = [f"{CIRCLES[c].replace(' ', '')}_BTS_{b + 1:03d}"
                    for c, b in zip(cust.circle, cust.bts)]
cust["cell"] = [f"404-45-{c * 10 + b + 100:03d}-{rng.integers(100, 999):03d}"
                for c, b in zip(cust.circle, cust.bts)]

activity = rng.lognormal(0, 0.5, N_CUSTOMERS)
activity /= activity.sum()

# Per-customer clock: the global curve shifted and sharpened per person, so an
# early riser and a night worker have different "normal" hours.
HOUR_W_CUST = np.array([np.roll(HOUR_W, s) ** k for s, k in
                        zip(rng.integers(-3, 4, N_CUSTOMERS), rng.uniform(1.0, 2.2, N_CUSTOMERS))])
HOUR_W_CUST /= HOUR_W_CUST.sum(1, keepdims=True)
HOUR_CW = HOUR_W_CUST.cumsum(1)
RAREST = np.argsort(HOUR_W_CUST, axis=1)   # each customer's least-likely hours first

# Persistent payee sets -- the fix for the 99.8% new-beneficiary baseline.
payee_n = rng.integers(3, 9, N_CUSTOMERS)
payee_flat, payee_off = ragged(payee_n, np.arange(N_CUSTOMERS))
contact_n = rng.integers(5, 13, N_CUSTOMERS)
contact_flat, contact_off = ragged(contact_n, np.arange(N_CUSTOMERS))

EXT_PHONES = np.char.add("+91", digits(3000, 10, first="6789"))
DEST_IPS = np.array([f"198.51.100.{i}" for i in range(1, 254)])
dest_n = rng.integers(3, 9, N_CUSTOMERS)
dest_flat, dest_off = ragged(dest_n, DEST_IPS)

# Mule accounts: receive only anomalous funds. Kept small and partly recycled so
# "novel receiver" is a strong hint, not a perfect give-away.
MULES = pd.DataFrame({
    "cid": (100900000 + np.arange(60)).astype(str),
    "name": [f"{a} {b}" for a, b in zip(rng.choice(FIRST, 60), rng.choice(LAST, 60))],
    "acct": digits(60, 12, first="123456789"),
    "phone": np.char.add("+91", digits(60, 10, first="6789")),
})
mule_bank = rng.choice(len(BANKS), 60)
MULES["bank"] = [BANKS[i][0] for i in mule_bank]
MULES["ifsc"] = np.char.add([BANKS[i][1] for i in mule_bank], digits(60, 7))

# --------------------------------------------------------------------------
# Baseline bank transactions
# --------------------------------------------------------------------------
snd = rng.choice(N_CUSTOMERS, N_BANK, p=activity)
rcv = pick(payee_flat, payee_off, payee_n, snd)
fresh = rng.random(N_BANK) < 0.08          # ~8% genuinely new payees
rcv[fresh] = rng.choice(N_CUSTOMERS, fresh.sum())
same = rcv == snd
rcv[same] = (rcv[same] + 1) % N_CUSTOMERS

dt = np.sort(stamps(N_BANK, owner=snd))
amount = np.round(np.exp(rng.normal(cust.mu.values[snd], cust.sigma.values[snd])), 2)
mode = rng.choice(MODES, N_BANK, p=MODE_W)
tid = np.where(mode == "ATM", "ATM", "TXN")
tid = np.char.add(np.char.add(tid, pd.Series(dt).dt.strftime("%y%m%d").values.astype("<U6")),
                  codes(N_BANK, 6))

S, R = cust.iloc[snd].reset_index(drop=True), cust.iloc[rcv].reset_index(drop=True)
bank = pd.DataFrame({
    "Transaction_ID": tid, "Date": dcol(dt), "Timestamp": tcol(dt),
    "Txn_Ref_Number": codes(N_BANK, 12), "Transaction_Mode": mode, "Currency": "INR",
    "Transaction_Amount": amount,
    "Sender_Customer_ID": S.cid, "Sender_Customer_Name": S.name, "Sender_Bank_Name": S.bank,
    "Sender_Account_Number": S.acct, "Sender_Account_Type": S.acct_type,
    "Sender_IFSC": S.ifsc, "Sender_Phone_Number": S.phone,
    "Receiver_Customer_ID": R.cid, "Receiver_Customer_Name": R.name, "Receiver_Bank_Name": R.bank,
    "Receiver_Account_Number": R.acct, "Receiver_Account_Type": R.acct_type,
    "Receiver_IFSC": R.ifsc, "Receiver_Phone_Number": R.phone,
})
bank["_snd"] = snd
bank["_dt"] = dt
bank = bank.drop_duplicates("Transaction_ID").reset_index(drop=True)

# --------------------------------------------------------------------------
# Baseline CDR / IPDR
# --------------------------------------------------------------------------
csub = rng.choice(N_CUSTOMERS, N_CDR, p=activity)
peer = pick(contact_flat, contact_off, contact_n, csub)
b_party = cust.phone.values[peer]
ext = rng.random(N_CDR) < 0.15
b_party[ext] = rng.choice(EXT_PHONES, ext.sum())
ctype = rng.choice(["VOICE", "SMS", "MISSED"], N_CDR, p=[.79, .15, .06])
dur = np.clip(rng.lognormal(3.9, 0.95, N_CDR), 1, 3600).round()
dur[ctype != "VOICE"] = 0
cdt = np.sort(stamps(N_CDR, owner=csub))
roam = rng.random(N_CDR) < 0.04
ccircle = cust.circle.values[csub].copy()
ccircle[roam] = rng.choice(len(CIRCLES), roam.sum())
cbts = cust.bts.values[csub].copy()
away = (rng.random(N_CDR) < 0.10) | roam
cbts[away] = rng.integers(0, BTS_PER_CIRCLE, away.sum())

cdr = pd.DataFrame({
    "CDR_ID": [f"CDR2026{i:08d}" for i in range(len(cdt))],
    "Call_Date": dcol(cdt), "Call_Start_Time": tcol(cdt),
    "A_Party_Number": cust.phone.values[csub], "B_Party_Number": b_party,
    "Call_Type": ctype, "Call_Duration_Seconds": dur.astype(int),
    "IMSI": cust.imsi.values[csub], "IMEI": cust.imei.values[csub],
    "First_BTS_Location": [f"{CIRCLES[c].replace(' ', '')}_BTS_{b + 1:03d}" for c, b in zip(ccircle, cbts)],
    "First_Cell_Global_ID": [f"404-45-{c * 10 + b + 100:03d}-{(c * 37 + b * 11) % 900 + 99:03d}"
                             for c, b in zip(ccircle, cbts)],
    "Roaming_Network_Circle": [CIRCLES[c] for c in ccircle],
})

isub = rng.choice(N_CUSTOMERS, N_IPDR, p=activity)
idt = np.sort(stamps(N_IPDR, owner=isub))
ipdr = pd.DataFrame({
    "IPDR_ID": [f"IPDR2026{i:08d}" for i in range(N_IPDR)],
    "Session_Date": dcol(idt), "Session_Start_Time": tcol(idt),
    "Subscriber_IMSI": cust.imsi.values[isub], "Subscriber_MSISDN": cust.phone.values[isub],
    "Device_IMEI": cust.imei.values[isub],
    "Source_IP_Address": [f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
                          for _ in range(N_IPDR)],
    "Destination_IP_Address": pick(dest_flat, dest_off, dest_n, isub),
    "Destination_Port": rng.choice(PORTS, N_IPDR, p=PORT_W),
    "Cell_Global_ID": cust.cell.values[isub],
    "Session_Duration_Seconds": np.clip(rng.lognormal(2.5, 1.1, N_IPDR), 1, 7200).round().astype(int),
})

# --------------------------------------------------------------------------
# Anomaly injection
# --------------------------------------------------------------------------
SCENARIOS = ["ODD_HOUR_TRANSACTION", "CUSTOMER_RELATIVE_AMOUNT_SPIKE", "AMOUNT_VELOCITY_SPIKE",
             "TRANSACTION_BURST", "NEW_BENEFICIARY", "AMOUNT_PLUS_NEW_BENEFICIARY",
             "UNUSUAL_CALL_BEFORE_TRANSACTION", "CALL_THEN_HIGH_VALUE_TRANSFER",
             "CALL_THEN_NEW_BENEFICIARY", "REPEATED_CALLS_BEFORE_TRANSACTION",
             "NETWORK_SESSION_BURST_AROUND_TRANSACTION", "UNUSUAL_LOCATION_CONTEXT",
             "NEW_DEVICE_AROUND_TRANSACTION", "IMSI_IMEI_PAIR_NOVELTY",
             "SUBTLE_MULTI_SOURCE_SUSPICIOUS_PATTERN"]
# Injection strength by difficulty: EASY is unmistakable, HARD still clears the
# baseline noise floor but only just.
STRENGTH = {"EASY": 1.0, "MEDIUM": 0.6, "HARD": 0.33}

ban = bank.copy()
new_bank, new_cdr, new_ipdr, gt = [], [], [], []
cdr_seq, ipdr_seq = len(cdr), len(ipdr)
cust_med = np.exp(cust.mu.values)

anchors = rng.choice(len(ban), N_ANOMALY, replace=False)
scen_of = np.resize(SCENARIOS, N_ANOMALY)
diff_of = rng.choice(["EASY", "MEDIUM", "HARD"], N_ANOMALY, p=[.25, .35, .40])


def lerp(s, lo, hi):
    """Scale a magnitude between its HARD (lo) and EASY (hi) bound."""
    return lo + (hi - lo) * s


def add_cdr(c, when, dur_s, b_num=None, imei=None, circle=None):
    global cdr_seq
    row = cust.iloc[c]
    ci = row.circle if circle is None else circle
    b = rng.integers(0, BTS_PER_CIRCLE)
    new_cdr.append({
        "CDR_ID": f"CDR2026{cdr_seq:08d}", "Call_Date": when.strftime("%Y-%m-%d"),
        "Call_Start_Time": when.strftime("%H:%M:%S"), "A_Party_Number": row.phone,
        "B_Party_Number": b_num if b_num else rng.choice(EXT_PHONES),
        "Call_Type": "VOICE", "Call_Duration_Seconds": int(dur_s), "IMSI": row.imsi,
        "IMEI": imei if imei else row.imei,
        "First_BTS_Location": f"{CIRCLES[ci].replace(' ', '')}_BTS_{b + 1:03d}",
        "First_Cell_Global_ID": f"404-45-{ci * 10 + b + 100:03d}-{(ci * 37 + b * 11) % 900 + 99:03d}",
        "Roaming_Network_Circle": CIRCLES[ci],
    })
    cdr_seq += 1
    return new_cdr[-1]["CDR_ID"]


def add_ipdr(c, when, imei=None, cell=None, dur_s=None):
    global ipdr_seq
    row = cust.iloc[c]
    new_ipdr.append({
        "IPDR_ID": f"IPDR2026{ipdr_seq:08d}", "Session_Date": when.strftime("%Y-%m-%d"),
        "Session_Start_Time": when.strftime("%H:%M:%S"), "Subscriber_IMSI": row.imsi,
        "Subscriber_MSISDN": row.phone, "Device_IMEI": imei if imei else row.imei,
        "Source_IP_Address": f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}",
        "Destination_IP_Address": rng.choice(DEST_IPS),
        "Destination_Port": int(rng.choice(PORTS, p=PORT_W)),
        "Cell_Global_ID": cell if cell else row.cell,
        "Session_Duration_Seconds": int(dur_s if dur_s else np.clip(rng.lognormal(2.5, 1.1), 1, 7200)),
    })
    ipdr_seq += 1
    return new_ipdr[-1]["IPDR_ID"]


def clone_txn(idx, when, amt):
    r = ban.loc[idx].to_dict()
    tid = ("ATM" if r["Transaction_Mode"] == "ATM" else "TXN") + \
          when.strftime("%y%m%d") + codes(1, 6)[0]
    r["Transaction_ID"] = tid
    r["Date"], r["Timestamp"] = when.strftime("%Y-%m-%d"), when.strftime("%H:%M:%S")
    r["Txn_Ref_Number"], r["Transaction_Amount"] = codes(1, 12)[0], round(float(amt), 2)
    r["_dt"] = when
    new_bank.append(r)
    return tid  # caller must collect this to emit a GT row per clone


def set_mule(idx):
    m = MULES.iloc[rng.integers(0, len(MULES))]
    for k, v in [("Receiver_Customer_ID", m.cid), ("Receiver_Customer_Name", m["name"]),
                 ("Receiver_Bank_Name", m.bank), ("Receiver_Account_Number", m.acct),
                 ("Receiver_Account_Type", "Savings"), ("Receiver_IFSC", m.ifsc),
                 ("Receiver_Phone_Number", m.phone)]:
        ban.at[idx, k] = v


for a_i, (idx, scen, diff) in enumerate(zip(anchors, scen_of, diff_of)):
    s = STRENGTH[diff]
    c = int(ban.at[idx, "_snd"])
    t = ban.at[idx, "_dt"].to_pydatetime() if hasattr(ban.at[idx, "_dt"], "to_pydatetime") \
        else ban.at[idx, "_dt"]
    med = cust_med[c]
    f = dict.fromkeys(["Bank", "CDR", "IPDR", "Amount", "Time", "Beneficiary", "Velocity",
                       "Call", "Device", "Location", "Network"], 0)
    cids, iids = [], []

    if scen == "ODD_HOUR_TRANSACTION":
        # pick from the hours this customer personally almost never uses
        h = int(rng.choice(RAREST[c][:3 if s > .8 else (5 if s > .5 else 8)]))
        t = t.replace(hour=h)
        ban.at[idx, "Timestamp"] = t.strftime("%H:%M:%S")
        ban.at[idx, "_dt"] = t
        f["Bank"] = f["Time"] = 1

    elif scen in ("CUSTOMER_RELATIVE_AMOUNT_SPIKE", "CALL_THEN_HIGH_VALUE_TRANSFER"):
        ban.at[idx, "Transaction_Amount"] = round(med * rng.uniform(lerp(s, 6, 25), lerp(s, 12, 50)), 2)
        f["Bank"] = f["Amount"] = 1
        if scen.startswith("CALL"):
            cids.append(add_cdr(c, t - timedelta(minutes=int(rng.integers(5, 25))),
                                rng.uniform(lerp(s, 180, 900), lerp(s, 400, 2400))))
            f["CDR"] = f["Call"] = 1

    elif scen in ("AMOUNT_VELOCITY_SPIKE", "TRANSACTION_BURST"):
        k = int(rng.integers(lerp(s, 4, 9), lerp(s, 6, 14)))
        hot = scen == "AMOUNT_VELOCITY_SPIKE"
        clone_ids = []
        for j in range(k):
            cid_clone = clone_txn(idx, t - timedelta(minutes=int(rng.integers(1, 58))),
                                  med * rng.uniform(3, 8) if hot else np.exp(rng.normal(cust.mu.values[c], cust.sigma.values[c])))
            clone_ids.append(cid_clone)
        if hot:
            ban.at[idx, "Transaction_Amount"] = round(med * rng.uniform(3, 8), 2)
            f["Amount"] = 1
        f["Bank"] = f["Velocity"] = 1
        # Each sibling is part of the same suspicious episode; label all positive.
        # HANDOFF.md §3.4: only the anchor was in GT before — that caused near-
        # identical feature vectors with opposite labels, suppressing velocity features.
        scope_clone = "BANK_ONLY"
        for clone_j, clone_tid in enumerate(clone_ids):
            gt.append({
                "Anomaly_ID": f"ANOM{a_i:06d}C{clone_j:02d}",
                "Customer_ID": cust.cid.values[c],
                "Transaction_ID": clone_tid,
                "CDR_IDs": "", "IPDR_IDs": "",
                "Scenario_Type": scen, "Difficulty": diff, "Source_Scope": scope_clone,
                **{f"{kk}_Anomaly": vv for kk, vv in f.items()},
                "Injected_Signals": ";".join(kk for kk, vv in f.items()
                                             if vv and kk not in ("Bank", "CDR", "IPDR")),
                "Is_Suspicious": 1,
            })

    elif scen in ("NEW_BENEFICIARY", "AMOUNT_PLUS_NEW_BENEFICIARY", "CALL_THEN_NEW_BENEFICIARY"):
        set_mule(idx)
        f["Bank"] = f["Beneficiary"] = 1
        if scen == "AMOUNT_PLUS_NEW_BENEFICIARY":
            ban.at[idx, "Transaction_Amount"] = round(med * rng.uniform(lerp(s, 5, 20), lerp(s, 10, 40)), 2)
            f["Amount"] = 1
        if scen == "CALL_THEN_NEW_BENEFICIARY":
            cids.append(add_cdr(c, t - timedelta(minutes=int(rng.integers(3, 20))),
                                rng.uniform(lerp(s, 150, 700), lerp(s, 350, 1800))))
            f["CDR"] = f["Call"] = 1

    elif scen == "UNUSUAL_CALL_BEFORE_TRANSACTION":
        cids.append(add_cdr(c, t - timedelta(minutes=int(rng.integers(5, 25))),
                            rng.uniform(lerp(s, 200, 900), lerp(s, 450, 2400))))
        f["CDR"] = f["Call"] = 1

    elif scen == "REPEATED_CALLS_BEFORE_TRANSACTION":
        caller = rng.choice(EXT_PHONES)
        for j in range(int(rng.integers(lerp(s, 3, 6), lerp(s, 5, 10)))):
            cids.append(add_cdr(c, t - timedelta(minutes=int(rng.integers(1, 15))),
                                rng.uniform(20, 200), b_num=caller))
        f["CDR"] = f["Call"] = 1

    elif scen == "NETWORK_SESSION_BURST_AROUND_TRANSACTION":
        for j in range(int(rng.integers(lerp(s, 6, 20), lerp(s, 11, 36)))):
            iids.append(add_ipdr(c, t + timedelta(minutes=int(rng.integers(-15, 15))),
                                 dur_s=rng.uniform(60, 900)))
        f["IPDR"] = f["Network"] = 1

    elif scen == "UNUSUAL_LOCATION_CONTEXT":
        far = int(rng.choice([x for x in range(len(CIRCLES)) if x != cust.circle.values[c]]))
        b = int(rng.integers(0, BTS_PER_CIRCLE))
        cids.append(add_cdr(c, t - timedelta(minutes=int(rng.integers(2, 30))),
                            rng.uniform(30, 300), circle=far))
        iids.append(add_ipdr(c, t + timedelta(minutes=int(rng.integers(-10, 10))),
                             cell=f"404-45-{far * 10 + b + 100:03d}-{rng.integers(100, 999):03d}"))
        f["CDR"] = f["IPDR"] = f["Location"] = 1

    elif scen in ("NEW_DEVICE_AROUND_TRANSACTION", "IMSI_IMEI_PAIR_NOVELTY"):
        novel = digits(1, 15, first="35")[0]
        lead = int(rng.integers(1, 20 if s > .5 else 40))
        cids.append(add_cdr(c, t - timedelta(hours=lead), rng.uniform(20, 300), imei=novel))
        if scen == "NEW_DEVICE_AROUND_TRANSACTION":
            iids.append(add_ipdr(c, t + timedelta(minutes=int(rng.integers(-20, 20))), imei=novel))
            f["IPDR"] = 1
        f["CDR"] = f["Device"] = 1

    else:  # SUBTLE_MULTI_SOURCE_SUSPICIOUS_PATTERN -- several weak signals at once
        ban.at[idx, "Transaction_Amount"] = round(med * rng.uniform(3, 6), 2)
        cids.append(add_cdr(c, t - timedelta(minutes=int(rng.integers(10, 40))), rng.uniform(150, 500)))
        for j in range(int(rng.integers(4, 9))):
            iids.append(add_ipdr(c, t + timedelta(minutes=int(rng.integers(-20, 20))), dur_s=rng.uniform(40, 400)))
        f.update(Bank=1, CDR=1, IPDR=1, Amount=1, Call=1, Network=1)

    scope = "BANK_CDR_IPDR" if f["IPDR"] else ("BANK_CDR" if f["CDR"] else "BANK_ONLY")
    gt.append({
        "Anomaly_ID": f"ANOM{a_i:06d}", "Customer_ID": cust.cid.values[c],
        "Transaction_ID": ban.at[idx, "Transaction_ID"],
        "CDR_IDs": ";".join(cids), "IPDR_IDs": ";".join(iids),
        "Scenario_Type": scen, "Difficulty": diff, "Source_Scope": scope,
        **{f"{k}_Anomaly": v for k, v in f.items()},
        "Injected_Signals": ";".join(k for k, v in f.items()
                                     if v and k not in ("Bank", "CDR", "IPDR")),
        "Is_Suspicious": 1,
    })

if new_bank:
    ban = pd.concat([ban, pd.DataFrame(new_bank)], ignore_index=True)
ban = ban.sort_values("_dt").reset_index(drop=True)
cdr_a = pd.concat([cdr, pd.DataFrame(new_cdr)], ignore_index=True) if new_cdr else cdr.copy()
ipdr_a = pd.concat([ipdr, pd.DataFrame(new_ipdr)], ignore_index=True) if new_ipdr else ipdr.copy()
for d, dc, tc in ((cdr_a, "Call_Date", "Call_Start_Time"), (ipdr_a, "Session_Date", "Session_Start_Time")):
    d.sort_values([dc, tc], inplace=True)
    d.reset_index(drop=True, inplace=True)
gt_df = pd.DataFrame(gt)

# --------------------------------------------------------------------------
# Correlation ground truth (tight windows: was 65% positive, i.e. useless)
# --------------------------------------------------------------------------
WIN = 900


def build_index(keys, times):
    order = np.argsort(keys, kind="stable")
    return keys[order], times[order], order


def bank_cdr_gt(b, c):
    ct = pd.to_datetime(c.Call_Date + " " + c.Call_Start_Time).values.astype("datetime64[s]").astype(np.int64)
    bt = pd.to_datetime(b.Date + " " + b.Timestamp).values.astype("datetime64[s]").astype(np.int64)
    idx = {}
    for role, col in (("A", "A_Party_Number"), ("B", "B_Party_Number")):
        k = c[col].values
        o = np.argsort(k, kind="stable")
        idx[role] = (k[o], ct[o], c.CDR_ID.values[o])
    rows = []
    for i in range(len(b)):
        best = None
        for phone, who in ((b.Sender_Phone_Number.values[i], "SENDER"),
                           (b.Receiver_Phone_Number.values[i], "RECEIVER")):
            for role, direction in (("A", "OUTGOING"), ("B", "INCOMING")):
                ks, ts, ids = idx[role]
                lo, hi = np.searchsorted(ks, phone), np.searchsorted(ks, phone, "right")
                if lo == hi:
                    continue
                d = ts[lo:hi] - bt[i]
                j = np.argmin(np.abs(d))
                if abs(d[j]) <= WIN and (best is None or abs(d[j]) < abs(best[1])):
                    best = (ids[lo + j], int(d[j]), f"{who}_{direction}_ACTIVITY")
        if best is None:
            rows.append((b.Transaction_ID.values[i], "", "NO_MATCH", "", 0))
        else:
            rows.append((b.Transaction_ID.values[i], best[0], best[2], best[1], 1))
    return pd.DataFrame(rows, columns=["Transaction_ID", "CDR_ID", "Relationship_Type",
                                       "Time_Difference_Seconds", "Is_Correlated"])


def cdr_ipdr_gt(c, ip):
    ct = pd.to_datetime(c.Call_Date + " " + c.Call_Start_Time).values.astype("datetime64[s]").astype(np.int64)
    it = pd.to_datetime(ip.Session_Date + " " + ip.Session_Start_Time).values.astype("datetime64[s]").astype(np.int64)
    k = ip.Subscriber_MSISDN.values
    o = np.argsort(k, kind="stable")
    ks, ts, ids = k[o], it[o], ip.IPDR_ID.values[o]
    rows = []
    for i in range(len(c)):
        lo, hi = np.searchsorted(ks, c.A_Party_Number.values[i]), np.searchsorted(ks, c.A_Party_Number.values[i], "right")
        hit = False
        if lo < hi:
            d = ts[lo:hi] - ct[i]
            near = np.argsort(np.abs(d))[:2]
            for rank, j in enumerate(near):
                if abs(d[j]) <= WIN:
                    rows.append((c.CDR_ID.values[i], ids[lo + j],
                                 "PRIMARY_SESSION" if rank == 0 else "ADDITIONAL_SESSION", int(d[j]), 1))
                    hit = True
        if not hit:
            rows.append((c.CDR_ID.values[i], "", "NO_MATCH", "", 0))
    return pd.DataFrame(rows, columns=["CDR_ID", "IPDR_ID", "Relationship_Type",
                                       "Time_Difference_Seconds", "Is_Correlated"])


# --------------------------------------------------------------------------
# Write + self-validate
# --------------------------------------------------------------------------
for sub in ("clean", "anomalous", "ground_truth"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
drop = ["_snd", "_dt"]
bank.drop(columns=drop).to_csv(OUT / "clean/bank_final.csv", index=False)
cdr.to_csv(OUT / "clean/cdr_final.csv", index=False)
ipdr.to_csv(OUT / "clean/ipdr_final.csv", index=False)
ban.drop(columns=drop).to_csv(OUT / "anomalous/bank_anomaly.csv", index=False)
cdr_a.to_csv(OUT / "anomalous/cdr_anomaly.csv", index=False)
ipdr_a.to_csv(OUT / "anomalous/ipdr_anomaly.csv", index=False)
gt_df.to_csv(OUT / "ground_truth/anomaly_ground_truth.csv", index=False)
bank_cdr_gt(ban, cdr_a).to_csv(OUT / "ground_truth/bank_cdr_ground_truth.csv", index=False)
cdr_ipdr_gt(cdr_a, ipdr_a).to_csv(OUT / "ground_truth/cdr_ipdr_ground_truth.csv", index=False)

# The check that would have caught the original dataset before any training ran.
lab = ban.Transaction_ID.isin(set(gt_df.Transaction_ID)).values.astype(int)
hour = pd.to_datetime(ban.Timestamp, format="%H:%M:%S").dt.hour.values
g = ban.groupby("Sender_Customer_ID").Transaction_Amount
z = ((ban.Transaction_Amount - g.transform("mean")) / (g.transform("std") + 1e-9)).fillna(0).values
newben = (~ban.duplicated(["Sender_Customer_ID", "Receiver_Account_Number"])).values.astype(int)

print(f"bank {len(ban):,} | cdr {len(cdr_a):,} | ipdr {len(ipdr_a):,} | "
      f"anomalies {lab.sum():,} ({lab.mean() * 100:.2f}%)")
print(f"odd-hour(0-5) base rate : {(hour[lab == 0] < 6).mean():.3f}  (was 0.249)")
print(f"new-beneficiary base    : {newben[lab == 0].mean():.3f}  (was 0.998)")
print(f"amount z>3: base {(z[lab == 0] > 3).mean():.4f} | anom {(z[lab == 1] > 3).mean():.4f}  (was 0.0186 / 0.0000)")

changed = len(set(ban.Transaction_ID) - set(bank.Transaction_ID))
modified = (~ban.set_index("Transaction_ID").reindex(bank.Transaction_ID)
            .Transaction_Amount.eq(bank.set_index("Transaction_ID").Transaction_Amount)).sum()
assert (z[lab == 1] > 3).mean() > (z[lab == 0] > 3).mean(), "amount signal still inverted"
assert (hour[lab == 0] < 6).mean() < 0.06, "no diurnal structure in baseline"
assert 0.03 < newben[lab == 0].mean() < 0.20, "beneficiary baseline unrealistic"
print(f"[OK] injected rows added={changed} amount-modified={modified}")
