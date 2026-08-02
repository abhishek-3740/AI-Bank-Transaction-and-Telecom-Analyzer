#!/usr/bin/env python
"""Shared causal feature builder for TRI-NETRA Stage 7.

Imported by BOTH scripts/train.py and scripts/score.py so exactly one feature
implementation exists in the repo (see HANDOFF.md section 3.2 -- a third copy
is how this codebase got into trouble the first time).

Every feature uses only events strictly before the anchor transaction's
timestamp. Use index_by()/before() for any new lookup rather than writing one.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
DAY = 86400

# Read every join key as str: pandas parses "+919812345678" as int64 otherwise,
# which silently breaks phone matching between the three sources.
STR = dict.fromkeys(["Sender_Phone_Number", "Receiver_Phone_Number", "Sender_Account_Number",
                     "Receiver_Account_Number", "Sender_Customer_ID", "Receiver_Customer_ID",
                     "A_Party_Number", "B_Party_Number", "IMSI", "IMEI",
                     "Subscriber_IMSI", "Subscriber_MSISDN", "Device_IMEI"], str)


def load_sources(variant="anomalous"):
    """Load bank/cdr/ipdr (+ ground truth) with epoch-second `ts` columns."""
    names = {"anomalous": ("anomalous/bank_anomaly.csv", "anomalous/cdr_anomaly.csv",
                           "anomalous/ipdr_anomaly.csv"),
             "clean": ("clean/bank_final.csv", "clean/cdr_final.csv", "clean/ipdr_final.csv")}[variant]
    bank, cdr, ipdr = (pd.read_csv(D / n, dtype=STR) for n in names)
    gt = pd.read_csv(D / "ground_truth/anomaly_ground_truth.csv")
    for df, dc, tc in ((bank, "Date", "Timestamp"), (cdr, "Call_Date", "Call_Start_Time"),
                       (ipdr, "Session_Date", "Session_Start_Time")):
        df["ts"] = pd.to_datetime(df[dc] + " " + df[tc]).astype("int64") // 10**9
    bank = bank.sort_values("ts").reset_index(drop=True)
    bank["y"] = bank.Transaction_ID.isin(set(gt.Transaction_ID)).astype(int)
    return bank, cdr, ipdr, gt

def first_seen(frame, col):
    """Earliest epoch-second each value of `col` appears."""
    return frame.groupby(col).ts.min().to_dict()


def index_by(frame, key):
    """key -> (sorted timestamps, positional index) for causal window lookups."""
    o = np.argsort(frame[key].values, kind="stable")
    k, t = frame[key].values[o], frame.ts.values[o]
    uniq, start = np.unique(k, return_index=True)
    end = np.append(start[1:], len(k))
    return {u: (t[s:e], o[s:e]) for u, s, e in zip(uniq, start, end)}


def before(idx, key, t, window=None):
    """Positions of events for `key` strictly before t (optionally within window)."""
    ent = idx.get(key)
    if ent is None:
        return np.array([], int), np.array([], int)
    ts, pos = ent
    hi = np.searchsorted(ts, t)
    lo = np.searchsorted(ts, t - window) if window else 0
    return ts[lo:hi], pos[lo:hi]


def build_features(bank, cdr, ipdr, verbose=True):
    """Return (bank_frame, {"A": df, "B": df, "C": df}) of causal features."""
    # ---------------------------------------------------------------- Set A: bank
    if verbose: print("building bank features ...")
    b = bank
    g = b.groupby("Sender_Customer_ID", sort=False)
    A = pd.DataFrame(index=b.index)
    A["transaction_amount"] = b.Transaction_Amount
    A["transaction_hour"] = pd.to_datetime(b.Timestamp, format="%H:%M:%S").dt.hour
    A["customer_history_count"] = g.cumcount()

    past_med = g.Transaction_Amount.apply(lambda s: s.shift().expanding().median()).reset_index(level=0, drop=True)
    past_mad = g.Transaction_Amount.apply(
        lambda s: s.shift().expanding().apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    ).reset_index(level=0, drop=True)
    A["amount_vs_customer_median"] = b.Transaction_Amount / (past_med + 1e-6)
    A["amount_robust_zscore"] = (b.Transaction_Amount - past_med) / (1.4826 * past_mad + 1e-6)
    A["amount_percentile"] = g.Transaction_Amount.apply(
        lambda s: s.shift().expanding().apply(lambda w: (w < w[-1]).mean() if len(w) else np.nan, raw=True)
    ).reset_index(level=0, drop=True)

    pair = b.Sender_Customer_ID.astype(str) + "|" + b.Receiver_Account_Number.astype(str)
    A["receiver_seen_before"] = pair.duplicated().astype(int)
    A["receiver_historical_count"] = b.groupby("Receiver_Account_Number", sort=False).cumcount()
    A["receiver_frequency"] = pair.groupby(pair).cumcount() / (A.customer_history_count + 1)
    # Laplace-smoothed share of the customer's prior activity in this 6h bucket.
    # Per-hour bins are unusable here: ~52 transactions cannot fill 24 of them.
    bucket = b.Sender_Customer_ID.astype(str) + "|" + (A.transaction_hour // 6).astype(str)
    A["hour_rarity"] = 1 - (bucket.groupby(bucket).cumcount() + 1) / (A.customer_history_count + 4)

    # Velocity via the same causal-window helper as CDR/IPDR. groupby().rolling()
    # returns rows in group order, so assigning its .values back scrambles the
    # alignment -- this keeps one mechanism that is positionally explicit.
    snd_idx = index_by(b, "Sender_Customer_ID")
    amt = b.Transaction_Amount.values
    # 10m/30m/1h/2h/6h counts (5) + amount sums for 30m/1h/2h/6h (4) +
    # 7d count+sum (2) + max amount in 1h/2h (2) = 13 cols
    vel = np.zeros((len(b), 13))
    for i, (cid, t) in enumerate(zip(b.Sender_Customer_ID.values, b.ts.values)):
        ts_all, pos_all = before(snd_idx, cid, t)
        for j, w in enumerate((600, 1800, 3600, 7200, 21600)):
            sel = pos_all[ts_all >= t - w]
            vel[i, j] = len(sel)
            if j >= 1:  # amount sums for 30m, 1h, 2h, 6h
                vel[i, 4 + j] = amt[sel].sum()
            if j == 2 and len(sel):  # max amount in 1h
                vel[i, 11] = amt[sel].max()
            if j == 3 and len(sel):  # max amount in 2h
                vel[i, 12] = amt[sel].max()
        # 7-day count and amount sum for ratio feature
        sel_7d = pos_all[ts_all >= t - 7 * DAY]
        vel[i, 9] = len(sel_7d)
        vel[i, 10] = amt[sel_7d].sum() if len(sel_7d) else 0
    for j, lbl in enumerate(("10m", "30m", "1h", "2h", "6h")):
        A[f"txn_count_previous_{lbl}"] = vel[:, j]
    A["amount_velocity_30m"] = vel[:, 5]
    A["amount_velocity_1h"] = vel[:, 6]
    A["amount_velocity_2h"] = vel[:, 7]
    A["amount_velocity_6h"] = vel[:, 8]
    # Per-customer normalised burst magnitude: window amount / 7-day mean amount
    # When no 7d history exists, set ratio to NaN (imputed later) instead of huge outlier
    with np.errstate(invalid="ignore"):
        mean_7d = np.where(vel[:, 9] > 0, vel[:, 10] / vel[:, 9], np.nan)
    A["amount_ratio_30m_to_7d"] = vel[:, 5] / np.where(np.isnan(mean_7d), np.nan, mean_7d + 1e-6)
    A["amount_ratio_1h_to_7d"] = vel[:, 6] / np.where(np.isnan(mean_7d), np.nan, mean_7d + 1e-6)
    A["amount_ratio_2h_to_7d"] = vel[:, 7] / np.where(np.isnan(mean_7d), np.nan, mean_7d + 1e-6)
    # Burst acceleration: how many times denser is the last 1h compared to the 6h baseline?
    rate_6h = vel[:, 4] / 6.0  # avg txns per hour over 6h
    A["txn_rate_acceleration"] = vel[:, 2] / (rate_6h + 0.1)
    # Per-customer normalized velocity: 1h count / customer's average daily txn rate.
    # A burst of 8 txns in 1h when customer averages 0.5/day produces ratio ~16.
    days_active = (b.ts.values - b.groupby("Sender_Customer_ID").ts.transform("min").values) / DAY + 1
    daily_rate = (A["customer_history_count"] + 1) / days_active
    A["txn_velocity_vs_customer_norm"] = vel[:, 2] / (daily_rate + 0.01)
    A["time_since_previous_transaction"] = g.ts.diff()

    # ---------------------------------------------------------------- Set B: +CDR
    if verbose: print("building cdr features ...")
    cdr_a = index_by(cdr, "A_Party_Number")
    cdr_b = index_by(cdr, "B_Party_Number")
    imei_fs, cell_fs = first_seen(cdr, "IMEI"), first_seen(cdr, "First_Cell_Global_ID")
    cdr["pair"] = cdr.IMSI + "_" + cdr.IMEI
    pair_fs = first_seen(cdr, "pair")
    peer_cnt = cdr.groupby(["A_Party_Number", "B_Party_Number"]).size().to_dict()
    home_circle_map = cdr.groupby("A_Party_Number").Roaming_Network_Circle.agg(lambda s: s.mode().iat[0]).to_dict()

    B = np.zeros((len(b), 16))
    dur, cimei, ccell, croam, cpeer = (cdr.Call_Duration_Seconds.values, cdr.IMEI.values,
                                       cdr.First_Cell_Global_ID.values, cdr.Roaming_Network_Circle.values,
                                       cdr.B_Party_Number.values)
    cpair = cdr["pair"].values
    for i, (phone, t) in enumerate(zip(b.Sender_Phone_Number.values, b.ts.values)):
        ts_all, pos_all = before(cdr_a, phone, t)
        B[i, 0] = len(pos_all) > 0
        for j, w in enumerate((600, 1800, 3600)):
            B[i, 1 + j] = (ts_all >= t - w).sum()
        if len(pos_all) == 0:
            B[i, 4] = np.nan
            continue
        B[i, 4] = t - ts_all[-1]
        w30 = pos_all[ts_all >= t - 1800]
        B[i, 5] = dur[w30].sum()
        B[i, 6] = dur[w30].max() if len(w30) else 0
        last = pos_all[-1]
        B[i, 7] = peer_cnt.get((phone, cpeer[last]), 0) <= 1          # caller_novelty
        B[i, 8] = peer_cnt.get((phone, cpeer[last]), 0) / len(pos_all)  # caller_historical_frequency
        # Device/cell novelty over the whole preceding day: a device swap shows up in
        # some call near the transaction, not necessarily the immediately previous one.
        w24 = pos_all[ts_all >= t - DAY]
        B[i, 9] = any(0 <= t - imei_fs.get(cimei[p], 0) < 7 * DAY for p in w24)
        B[i, 10] = any(0 <= t - cell_fs.get(ccell[p], 0) < 7 * DAY for p in w24)
        B[i, 11] = any(0 <= t - pair_fs.get(cpair[p], 0) < 7 * DAY for p in w24)
        # Location features: distinct circles in 24h and circle != home circle
        B[i, 12] = len(set(croam[p] for p in w24))  # distinct_circles_24h
        hc = home_circle_map.get(phone)
        B[i, 13] = any(croam[p] != hc for p in w24) if hc else 0  # circle_mismatch_24h
        # Tighter 30m location window — more targeted for the injection pattern
        w30loc = pos_all[ts_all >= t - 1800]
        B[i, 14] = len(set(croam[p] for p in w30loc))  # distinct_circles_30m
        B[i, 15] = any(croam[p] != hc for p in w30loc) if (hc and len(w30loc)) else 0  # circle_mismatch_30m
    Bdf = pd.DataFrame(B, columns=["has_cdr_context", "calls_previous_10m", "calls_previous_30m",
                                   "calls_previous_1h", "nearest_call_before_seconds",
                                   "total_call_duration_30m", "max_call_duration_30m",
                                   "caller_novelty", "caller_historical_frequency",
                                   "imei_novelty", "cell_novelty", "cdr_imsi_imei_pair_novelty",
                                   "distinct_circles_24h", "circle_mismatch_24h",
                                   "distinct_circles_30m", "circle_mismatch_30m"])
    # roaming_change: last call outside the customer's modal circle
    rc = np.zeros(len(b))
    for i, (phone, t) in enumerate(zip(b.Sender_Phone_Number.values, b.ts.values)):
        _, pos = before(cdr_a, phone, t)
        rc[i] = croam[pos[-1]] != home_circle_map.get(phone) if len(pos) else 0
    Bdf["roaming_change"] = rc

    # --------------------------------------------------------------- Set C: +IPDR
    if verbose: print("building ipdr features ...")
    ip_idx = index_by(ipdr, "Subscriber_MSISDN")
    sip_fs, dip_fs = first_seen(ipdr, "Source_IP_Address"), first_seen(ipdr, "Destination_IP_Address")
    ipdr["pport"] = ipdr.Subscriber_MSISDN + "|" + ipdr.Destination_Port.astype(str)
    ipdr["ppair"] = ipdr.Subscriber_IMSI + "_" + ipdr.Device_IMEI
    port_fs, ipair_fs = first_seen(ipdr, "pport"), first_seen(ipdr, "ppair")
    isip, idip, iport = (ipdr.Source_IP_Address.values, ipdr.Destination_IP_Address.values,
                         ipdr["pport"].values)
    iimei, icell, isdur = ipdr.Device_IMEI.values, ipdr.Cell_Global_ID.values, ipdr.Session_Duration_Seconds.values
    ippair = ipdr["ppair"].values
    sub_med = ipdr.groupby("Subscriber_MSISDN").Session_Duration_Seconds.median().to_dict()
    sub_imei = ipdr.groupby("Subscriber_MSISDN").Device_IMEI.agg(lambda s: s.mode().iat[0]).to_dict()
    sub_cell = ipdr.groupby("Subscriber_MSISDN").Cell_Global_ID.agg(lambda s: s.mode().iat[0]).to_dict()

    C = np.zeros((len(b), 11))
    for i, (phone, t) in enumerate(zip(b.Sender_Phone_Number.values, b.ts.values)):
        ts_all, pos_all = before(ip_idx, phone, t)
        C[i, 0] = len(pos_all) > 0
        C[i, 1] = (ts_all >= t - 600).sum()
        C[i, 2] = (ts_all >= t - 1800).sum()
        if len(pos_all) == 0:
            C[i, 3] = np.nan
            continue
        C[i, 3] = t - ts_all[-1]
        last, lt = pos_all[-1], ts_all[-1]
        C[i, 4] = (lt - sip_fs.get(isip[last], 0)) < DAY
        C[i, 5] = (lt - dip_fs.get(idip[last], 0)) < 7 * DAY
        C[i, 6] = (lt - port_fs.get(iport[last], 0)) < 7 * DAY
        C[i, 7] = (lt - ipair_fs.get(ippair[last], 0)) < 7 * DAY
        C[i, 8] = iimei[last] == sub_imei.get(phone)
        C[i, 9] = icell[last] == sub_cell.get(phone)
        C[i, 10] = isdur[last] / (sub_med.get(phone, 1) + 1e-6)
    Cdf = pd.DataFrame(C, columns=["has_ipdr_context", "sessions_previous_10m", "sessions_previous_30m",
                                   "nearest_session_before_seconds", "source_ip_novelty",
                                   "destination_ip_novelty", "destination_port_novelty",
                                   "imsi_imei_pair_novelty", "device_consistency",
                                   "cell_consistency", "session_duration_deviation"])

    SETS = {"A": A, "B": pd.concat([A, Bdf], axis=1), "C": pd.concat([A, Bdf, Cdf], axis=1)}
    return b, SETS
