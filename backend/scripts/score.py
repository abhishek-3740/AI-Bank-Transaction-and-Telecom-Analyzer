#!/usr/bin/env python
"""Stage 8: score transactions with the trained Stage 7 model.

Turns the raw model probability into an investigator-facing artefact: a 0-100
risk score, a band, the three features that actually drove the score (exact
TreeSHAP contributions, not feature_importances_), and any rules that fired.

The scoring itself lives in scoring_core.py so the API's upload path produces
identical scores; this script is the offline batch driver plus its report.

Run: python scripts/train.py && python scripts/score.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_sources, ROOT
from scoring_core import RULES, load_bundle, score_frame

OUT = ROOT / "notebook" / "output"

bundle = load_bundle()
print(f"model: {bundle['kind']}  val PR-AUC={bundle['val_pr_auc']:.4f}  "
      f"test PR-AUC={bundle['test_pr_auc']:.4f}  threshold={bundle['threshold']:.4f}")

bank, cdr, ipdr, gt = load_sources()
rep = score_frame(bank, cdr, ipdr, bundle=bundle)

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
for rule_name, _ in RULES:
    fired = rep.rules_fired.str.contains(rule_name, regex=False)
    print(f"  {rule_name:<22s}  fired={int(fired.sum()):,}  "
          f"true_positives={int(rep.loc[fired, 'is_suspicious_gt'].sum())}")
print("\ntop 10 alerts (held-out rows):")
print(rep[te].head(10)[["Transaction_ID", "Transaction_Amount", "risk_score", "risk_band",
                        "reason_1", "is_suspicious_gt"]].to_string(index=False))
print(f"\nwrote {OUT / 'scored_transactions.csv'}")
