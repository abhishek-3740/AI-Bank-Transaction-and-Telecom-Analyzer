#!/usr/bin/env python
"""Stage 8: score transactions with the trained Stage 7 model.

Turns the raw model probability into an investigator-facing artefact: a 0-100
risk score, a band, the three features that actually drove the score (exact
TreeSHAP contributions, not feature_importances_), and any rules that fired.

Consumes the bundle written by scripts/train.py -- it never retrains, and it
reuses that bundle's imputation medians, because refitting them here would leak
and silently shift every score.

Run: python scripts/train.py && python scripts/score.py
"""
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_sources, build_features, ROOT

OUT = ROOT / "notebook" / "output"
BUNDLE = ROOT / "models" / "stage7_setC.joblib"

if not BUNDLE.exists():
    raise SystemExit(f"{BUNDLE} not found -- run: python scripts/train.py")
bundle = joblib.load(BUNDLE)
if bundle["kind"] != "XGBoost":
    raise NotImplementedError(
        f"bundle holds a '{bundle['kind']}' model; only the plain XGBoost path is "
        "implemented here. Extend score.py before switching the selected model.")

print(f"model: {bundle['kind']}  val PR-AUC={bundle['val_pr_auc']:.4f}  "
      f"test PR-AUC={bundle['test_pr_auc']:.4f}  threshold={bundle['threshold']:.4f}")

bank, cdr, ipdr, gt = load_sources()
b, SETS = build_features(bank, cdr, ipdr)
X = SETS["C"].replace([np.inf, -np.inf], np.nan)[bundle["columns"]]
X = X.fillna(bundle["medians"]).fillna(0)

model, thr = bundle["model"], bundle["threshold"]
prob = model.predict_proba(X)[:, 1]

# Same temporal boundaries as training, so the report can mark which rows the
# model actually held out. Scores on train-period rows are in-sample.
cut_tr, cut_va = np.quantile(b.ts, 0.60), np.quantile(b.ts, 0.75)
split = np.where(b.ts < cut_tr, "train", np.where(b.ts < cut_va, "val", "test"))

# Piecewise-linear map pinning the tuned threshold to 70, so "risk >= 70" means
# "the model would alert" rather than an arbitrary cutoff.
risk = np.where(prob < thr, 70 * prob / thr, 70 + 30 * (prob - thr) / (1 - thr + 1e-12))
risk = np.clip(risk, 0, 100).round(1)
band = pd.cut(risk, [-0.1, 40, 70, 90, 100], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

# Exact per-transaction TreeSHAP contributions; last column is the bias term.
contrib = model.get_booster().predict(xgb.DMatrix(X, feature_names=list(X.columns)),
                                      pred_contribs=True)[:, :-1]
top3 = np.argsort(-contrib, axis=1)[:, :3]
cols = np.array(X.columns)
reasons = [[f"{cols[j]} ({contrib[i, j]:+.2f})" if contrib[i, j] > 0 else ""
            for j in top3[i]] for i in range(len(X))]

# ------------------------------------------------------------------ Rule engine
# Rules fire on raw transaction/feature signals that the ML model cannot rank
# reliably on their own (HANDOFF.md §3.1).  Each rule is a boolean mask; the
# final column joins the names of all firing rules with '|'.
#
# Rules implemented:
#   ODD_HOUR           – transaction between 00:00–05:59 for a customer whose
#                        baseline night activity is near zero.  ML cannot rank
#                        this because 855 night txns / only 37 anomalous.
#   HIGH_AMOUNT_ANOMALY – transaction amount > 5× the customer's median.
#                        Catches CUSTOMER_RELATIVE_AMOUNT_SPIKE when the ML
#                        model is uncertain (low CDR/IPDR context).
#   RAPID_SUCCESSION   – ≥3 transactions in the previous 10 minutes.
#   NEW_BENEFICIARY_FLAG – first-ever transfer to this receiver account.
#   TELECOM_BURST      – ≥3 CDR calls in the 30 min window before the txn.

hour = X["transaction_hour"].values
r_odd_hour = (hour >= 0) & (hour <= 5)

amount_ratio = X["amount_vs_customer_median"].fillna(0).values
r_high_amount = amount_ratio > 5.0

rapid = X["txn_count_previous_10m"].values
r_rapid = rapid >= 3

# receiver_seen_before == 0 means this is the first time this sender sent to
# this receiver (the feature is 1 when seen before, 0 when novel).
r_new_ben = X["receiver_seen_before"].values == 0

calls_30m = X["calls_previous_30m"].values
r_telecom = calls_30m >= 3

# Build the combined rule string per transaction
rule_parts = [
    ("ODD_HOUR",            r_odd_hour),
    ("HIGH_AMOUNT_ANOMALY", r_high_amount),
    ("RAPID_SUCCESSION",    r_rapid),
    ("NEW_BENEFICIARY_FLAG",r_new_ben),
    ("TELECOM_BURST",       r_telecom),
]
rules_fired_list = []
for masks in zip(*[m for _, m in rule_parts]):
    names = [name for (name, _), fired in zip(rule_parts, masks) if fired]
    rules_fired_list.append("|".join(names))
rules = np.array(rules_fired_list)

rep = pd.DataFrame({
    "Transaction_ID": b.Transaction_ID, "Date": b.Date, "Timestamp": b.Timestamp,
    "Sender_Customer_ID": b.Sender_Customer_ID, "Sender_Customer_Name": b.Sender_Customer_Name,
    "Receiver_Account_Number": b.Receiver_Account_Number,
    "Transaction_Amount": b.Transaction_Amount,
    "ml_probability": prob.round(4), "risk_score": risk, "risk_band": band,
    "reason_1": [r[0] for r in reasons], "reason_2": [r[1] for r in reasons],
    "reason_3": [r[2] for r in reasons],
    "rules_fired": rules, "split": split, "is_suspicious_gt": b.y,
}).sort_values("risk_score", ascending=False).reset_index(drop=True)

OUT.mkdir(parents=True, exist_ok=True)
rep.to_csv(OUT / "scored_transactions.csv", index=False)

alerts = rep[rep.risk_score >= 70]
te = rep.split == "test"
print(f"\nscored {len(rep):,} transactions | {len(alerts):,} at risk>=70 "
      f"({len(alerts) / len(rep) * 100:.2f}%)")
print("\nband distribution:")
print(rep.risk_band.value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).to_string())
print(f"\nheld-out (test) rows only: {te.sum():,} transactions, {rep[te].is_suspicious_gt.sum()} true anomalies")
a_te = rep[te & (rep.risk_score >= 70)]
if len(a_te):
    print(f"  alerts={len(a_te)}  precision={a_te.is_suspicious_gt.mean():.3f}  "
          f"recall={a_te.is_suspicious_gt.sum() / max(rep[te].is_suspicious_gt.sum(), 1):.3f}")
print(f"\nrules fired: {(rep.rules_fired != '').sum():,} transactions carry at least one rule "
      f"({rep[rep.rules_fired != ''].is_suspicious_gt.sum()} of them true anomalies)")
for rule_name, mask in rule_parts:
    n = int(np.array([rule_name in r for r in rules_fired_list]).sum())
    tp = int(b.y.values[np.array([rule_name in r for r in rules_fired_list])].sum())
    print(f"  {rule_name:<22s}  fired={n:,}  true_positives={tp}")
print("\ntop 10 alerts (held-out rows):")
print(rep[te].head(10)[["Transaction_ID", "Transaction_Amount", "risk_score", "risk_band",
                        "reason_1", "is_suspicious_gt"]].to_string(index=False))
print(f"\nwrote {OUT / 'scored_transactions.csv'}")
