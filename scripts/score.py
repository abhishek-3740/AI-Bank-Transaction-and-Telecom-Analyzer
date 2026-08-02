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

# Rule engine. Deliberately minimal: ODD_HOUR is here because a lone odd-hour
# transaction is provably unrankable by the ML model (HANDOFF.md 3.1) -- 855
# night transactions, only 37 anomalous. It belongs in rules, not the ranker.
hour = X["transaction_hour"].values
rules = np.where((hour >= 0) & (hour <= 5), "ODD_HOUR", "")

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
print(f"\nrules fired: {(rep.rules_fired != '').sum():,} ODD_HOUR "
      f"({rep[rep.rules_fired != ''].is_suspicious_gt.sum()} of them true anomalies)")
print("\ntop 10 alerts (held-out rows):")
print(rep[te].head(10)[["Transaction_ID", "Transaction_Amount", "risk_score", "risk_band",
                        "reason_1", "is_suspicious_gt"]].to_string(index=False))
print(f"\nwrote {OUT / 'scored_transactions.csv'}")
