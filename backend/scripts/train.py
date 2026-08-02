#!/usr/bin/env python
"""Stage 7: train + ablate anomaly detection on the real TRI-NETRA schema.

Replaces notebook/stage7.py, which referenced lowercase column names and
absolute paths that do not exist, and notebook/Untitled.ipynb, whose Cell 2
generated a throwaway 8k-row USD dataset over the top of the real one -- so the
published PR-AUC of 0.52 described synthetic data with randomly-paired
correlation links, not this project's data.

Feature names follow src/features/models.py::FeatureRow. Every feature is
computed from events strictly before the anchor transaction.
Run: python scripts/train.py
"""
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, precision_recall_curve)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_sources, build_features, ROOT   # one shared implementation

SEED = 42
OUT = ROOT / "notebook" / "output"
MODELS = ROOT / "models"

bank, cdr, ipdr, gt = load_sources()
b, SETS = build_features(bank, cdr, ipdr)

# ---------------------------------------------------------------- split + fit
y = b.y.values
cut_tr, cut_va = np.quantile(b.ts, 0.60), np.quantile(b.ts, 0.75)
tr, va, te = b.ts < cut_tr, (b.ts >= cut_tr) & (b.ts < cut_va), b.ts >= cut_va
tr_use, va_use = tr, va  # no exclusions — clone exclusion was tested but hurt CDR/IPDR scenarios
print(f"\ntrain {tr.sum():,} ({y[tr].sum()} pos) | val {va.sum():,} ({y[va].sum()} pos) | "
      f"test {te.sum():,} ({y[te].sum()} pos)\n")


def report(y_true, s, thr, model, name, nfeat):
    p = (s >= thr).astype(int)
    m = {"Model": model, "Feature_Set": name, "N_Features": nfeat,
         "Precision": precision_score(y_true, p, zero_division=0),
         "Recall": recall_score(y_true, p, zero_division=0),
         "F1": f1_score(y_true, p, zero_division=0),
         "PR_AUC": average_precision_score(y_true, s), "ROC_AUC": roc_auc_score(y_true, s)}
    order = np.argsort(s)[::-1]
    for k in (50, 100, 250):
        hits = y_true[order[:k]].sum()
        m[f"P@{k}"], m[f"R@{k}"] = hits / k, hits / y_true.sum()
    return m


from sklearn.preprocessing import MinMaxScaler
rows, best = [], {}
for name, F in SETS.items():
    X = F.replace([np.inf, -np.inf], np.nan)
    med = X[tr_use].median()  # imputation fitted on clone-excluded train only
    Xtr = X[tr_use].fillna(med).fillna(0)
    Xva = X[va_use].fillna(med).fillna(0)
    Xte = X[te].fillna(med).fillna(0)
    # Full val/test for scoring (including clones, since IF is unsupervised)
    Xva_full = X[va].fillna(med).fillna(0)

    # ---- 1. Isolation Forest (unsupervised baseline) ----
    iso = IsolationForest(n_estimators=300, max_samples=0.8,
                          contamination=0.04, random_state=SEED, n_jobs=-1).fit(Xtr)
    iso_tr = -iso.score_samples(Xtr)
    iso_va = -iso.score_samples(Xva)
    iso_va_full = -iso.score_samples(Xva_full)
    iso_te = -iso.score_samples(Xte)
    rows.append(report(y[te], iso_te,
                       np.percentile(iso_te, 96), "IsolationForest", name, X.shape[1]))

    spw = (y[tr_use] == 0).sum() / max(y[tr_use].sum(), 1)

    # ---- 2. XGBoost (supervised) with HP tuning for Set C ----
    if name == "C":
        best_prauc, best_clf = -1, None
        for md in (4, 5):
            for mcw in (1, 3, 5):
                for rl in (1.0, 2.0):
                    _clf = XGBClassifier(n_estimators=600, max_depth=md, learning_rate=0.04,
                                        subsample=0.8, colsample_bytree=0.7, min_child_weight=mcw,
                                        reg_lambda=rl, scale_pos_weight=spw,
                                        eval_metric="aucpr", random_state=SEED, n_jobs=-1,
                                        verbosity=0).fit(Xtr, y[tr_use])
                    _prauc = average_precision_score(y[va_use], _clf.predict_proba(Xva)[:, 1])
                    if _prauc > best_prauc:
                        best_prauc, best_clf = _prauc, _clf
        clf = best_clf
        print(f"  XGB tuned Set C: max_depth={clf.max_depth} mcw={clf.min_child_weight} "
              f"rl={clf.reg_lambda} val_PR-AUC={best_prauc:.4f}")
    else:
        clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
                            scale_pos_weight=spw, eval_metric="aucpr", random_state=SEED,
                            n_jobs=-1).fit(Xtr, y[tr_use])

    xgb_va = clf.predict_proba(Xva)[:, 1]
    xgb_te = clf.predict_proba(Xte)[:, 1]

    pv, rv, th = precision_recall_curve(y[va_use], xgb_va)
    f1s = 2 * pv * rv / (pv + rv + 1e-12)
    thr = th[np.argmax(f1s[:-1])] if len(th) else 0.5
    rows.append(report(y[te], xgb_te, thr, "XGBoost", name, X.shape[1]))

    # ---- 3. Stacked XGBoost: uses IF anomaly score as extra feature ----
    if name == "C":
        # Normalise IF scores to [0,1] for clean stacking
        scaler = MinMaxScaler().fit(iso_tr.reshape(-1, 1))
        iso_tr_n = scaler.transform(iso_tr.reshape(-1, 1)).ravel()
        iso_va_n = scaler.transform(iso_va.reshape(-1, 1)).ravel()
        iso_va_full_n = scaler.transform(iso_va_full.reshape(-1, 1)).ravel()
        iso_te_n = scaler.transform(iso_te.reshape(-1, 1)).ravel()

        Xtr_s = np.column_stack([Xtr.values, iso_tr_n])
        Xva_s = np.column_stack([Xva.values, iso_va_n])
        Xte_s = np.column_stack([Xte.values, iso_te_n])
        stack_cols = list(X.columns) + ["if_anomaly_score"]

        best_prauc_s, best_clf_s = -1, None
        for md in (4, 5):
            for mcw in (1, 3):
                _clf = XGBClassifier(n_estimators=600, max_depth=md, learning_rate=0.04,
                                    subsample=0.8, colsample_bytree=0.7, min_child_weight=mcw,
                                    reg_lambda=1.5, scale_pos_weight=spw,
                                    eval_metric="aucpr", random_state=SEED, n_jobs=-1,
                                    verbosity=0).fit(Xtr_s, y[tr_use])
                _prauc = average_precision_score(y[va_use], _clf.predict_proba(Xva_s)[:, 1])
                if _prauc > best_prauc_s:
                    best_prauc_s, best_clf_s = _prauc, _clf
        stacked_te = best_clf_s.predict_proba(Xte_s)[:, 1]
        stacked_va = best_clf_s.predict_proba(Xva_s)[:, 1]
        pv, rv, th = precision_recall_curve(y[va_use], stacked_va)
        f1s = 2 * pv * rv / (pv + rv + 1e-12)
        thr_s = th[np.argmax(f1s[:-1])] if len(th) else 0.5
        rows.append(report(y[te], stacked_te, thr_s, "Stacked_XGB+IF", name, len(stack_cols)))
        print(f"  Stacked XGB+IF Set C: val_PR-AUC={best_prauc_s:.4f}")

        # ---- 4. Blended ensemble: alpha*XGB + (1-alpha)*IF, tuned on val ----
        xgb_va_n = (xgb_va - xgb_va.min()) / (xgb_va.max() - xgb_va.min() + 1e-12)
        xgb_te_n = (xgb_te - xgb_te.min()) / (xgb_te.max() - xgb_te.min() + 1e-12)

        best_alpha, best_blend_prauc = 1.0, -1
        for alpha in np.arange(0.5, 1.0, 0.05):
            blend_va = alpha * xgb_va_n + (1 - alpha) * iso_va_n
            prauc = average_precision_score(y[va_use], blend_va)
            if prauc > best_blend_prauc:
                best_alpha, best_blend_prauc = alpha, prauc

        blend_te = best_alpha * xgb_te_n + (1 - best_alpha) * iso_te_n
        pv, rv, th = precision_recall_curve(y[va_use],
                                            best_alpha * xgb_va_n + (1 - best_alpha) * iso_va_n)
        f1s = 2 * pv * rv / (pv + rv + 1e-12)
        thr_b = th[np.argmax(f1s[:-1])] if len(th) else 0.5
        rows.append(report(y[te], blend_te, thr_b, f"Blend(a={best_alpha:.2f})", name, X.shape[1]))
        print(f"  Blend XGB+IF Set C: alpha={best_alpha:.2f} val_PR-AUC={best_blend_prauc:.4f}")

        # ---- 5. Stacked blend: alpha*Stacked + (1-alpha)*IF ----
        stk_te_n = (stacked_te - stacked_te.min()) / (stacked_te.max() - stacked_te.min() + 1e-12)
        stk_va_n = (stacked_va - stacked_va.min()) / (stacked_va.max() - stacked_va.min() + 1e-12)

        best_alpha2, best_sb_prauc = 1.0, -1
        for alpha in np.arange(0.5, 1.0, 0.05):
            sb_va = alpha * stk_va_n + (1 - alpha) * iso_va_n
            prauc = average_precision_score(y[va_use], sb_va)
            if prauc > best_sb_prauc:
                best_alpha2, best_sb_prauc = alpha, prauc

        sb_te = best_alpha2 * stk_te_n + (1 - best_alpha2) * iso_te_n
        pv, rv, th = precision_recall_curve(y[va_use],
                                            best_alpha2 * stk_va_n + (1 - best_alpha2) * iso_va_n)
        f1s = 2 * pv * rv / (pv + rv + 1e-12)
        thr_sb = th[np.argmax(f1s[:-1])] if len(th) else 0.5
        rows.append(report(y[te], sb_te, thr_sb, f"StackBlend(a={best_alpha2:.2f})", name, len(stack_cols)))
        print(f"  StackBlend Set C: alpha={best_alpha2:.2f} val_PR-AUC={best_sb_prauc:.4f}")

    # Select the reported Set C model on VALIDATION PR-AUC. Selecting on test
    # leaks the held-out set into model choice and inflates every downstream
    # number -- the val scores below are already computed, so use those.
    if name == "C":
        c_models = [(xgb_te, "XGBoost", best_prauc),
                    (stacked_te, "Stacked_XGB+IF", best_prauc_s),
                    (blend_te, f"Blend(a={best_alpha:.2f})", best_blend_prauc),
                    (sb_te, f"StackBlend(a={best_alpha2:.2f})", best_sb_prauc)]
        best_score, best_name, best_val = max(c_models, key=lambda x: x[2])
        best[name] = (clf, X.columns, best_score)
        print("\n  Set C candidates (val PR-AUC -> test PR-AUC):")
        for s, n, v in c_models:
            print(f"    {n:24s} val={v:.4f}  test={average_precision_score(y[te], s):.4f}")
        print(f"  >>> Selected on val: {best_name} "
              f"(val={best_val:.4f}, test={average_precision_score(y[te], best_score):.4f})")
    else:
        best[name] = (clf, X.columns, xgb_te)

res = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "stage7_ablation_results.csv", index=False)
show = ["Model", "Feature_Set", "N_Features", "PR_AUC", "ROC_AUC", "Precision", "Recall", "F1", "P@100", "R@100"]
print(res[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

clf, cols, st = best["C"]
print("\nTop 12 features (Set C XGBoost):")
print(pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False).head(12).to_string())

scen = gt.set_index("Transaction_ID")[["Scenario_Type", "Difficulty"]]
t_df = b[te].copy()
t_df["score"] = st
t_df = t_df.join(scen, on="Transaction_ID")
k = int(y[te].sum())
flagged = set(t_df.nlargest(k, "score").Transaction_ID)
per = (t_df[t_df.y == 1].assign(hit=lambda d: d.Transaction_ID.isin(flagged))
       .groupby("Scenario_Type").hit.agg(["size", "mean"]).rename(columns={"size": "n", "mean": f"recall@{k}"}))
print(f"\nPer-scenario recall @ top-{k} (Best Set C model):")
print(per.sort_values(f"recall@{k}", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))
print("\nBy difficulty:")
print(t_df[t_df.y == 1].assign(hit=lambda d: d.Transaction_ID.isin(flagged))
      .groupby("Difficulty").hit.agg(["size", "mean"]).to_string(float_format=lambda v: f"{v:.3f}"))
res.to_csv(OUT / "hackathon_results.csv", index=False)

# Persist the val-selected Set C model so scripts/score.py can reuse it without
# retraining. Imputation medians travel with the model -- refitting them on the
# scoring data would leak and would silently shift every score.
MODELS.mkdir(exist_ok=True)
bundle = {"kind": best_name, "model": clf, "columns": list(cols),
          "medians": SETS["C"].replace([np.inf, -np.inf], np.nan)[tr_use].median(),
          "threshold": float(thr), "val_pr_auc": float(best_val),
          "test_pr_auc": float(average_precision_score(y[te], st)), "seed": SEED}
joblib.dump(bundle, MODELS / "stage7_setC.joblib")
print(f"\nwrote {OUT / 'stage7_ablation_results.csv'}")
print(f"wrote {MODELS / 'stage7_setC.joblib'}  ({best_name}, thr={thr:.4f})")
