# TRI-NETRA — Model Training Handoff

**Problem Statement:** ERH26_PS_03 · **Repo:** AI-Bank-Transaction-and-Telecom-Analyzer
**Last updated:** 2026-08-02 · **Branch:** `main`

> **Read this file end to end before touching anything.** It records why the
> dataset was rebuilt, what the numbers currently mean, and several traps that
> cost real debugging time. Sections 1–3 are context you need to avoid redoing
> broken work. Section 6 is your task list.

---

## 1. What was wrong, and what was fixed

The model was underperforming. The cause was **not** the model — it was two
separate data problems.

### 1.1 The published results were never trained on this project's data

`notebook/Untitled.ipynb` **Cell 2** (`CELL 2 (HACKATHON): SCENARIO-AWARE
SYNTHETIC DATA`) generated its own 8,000-row dataset and **wrote it over the
`data/` paths**. Everything downstream trained on that. The reported PR-AUC of
0.52 in `notebook/output/hackathon_results.csv` described a fake USD dataset
with `sample()`-paired correlation links — not the real INR dataset.

`notebook/stage7.py` never ran at all: it referenced lowercase column names
(`transaction_id`) that do not exist in the CSVs, and hardcoded
`C:\Users\Arpit Mishra\...` paths.

**Status: FIXED.** Cell 2 is now a markdown note. `scripts/train.py` replaces
`stage7.py`.

### 1.2 The real dataset had almost no learnable signal

Measured on the old `data/` with the canonical fraud features:

```
AUC[Transaction_Amount] = 0.600     AUC[new_beneficiary] = 0.501
AUC[amount_zscore]      = 0.577     AUC[velocity_1h]     = 0.499
AUC[calls_30m_before]   = 0.545     AUC[hour]            = 0.459  <- worse than random
```

Root causes, all in the data generator, none fixable by modelling:

| Defect | Evidence |
|---|---|
| No diurnal baseline | hourly counts flat (4,153–4,325 across all 24h); 24.9% of *normal* txns were 00:00–06:00 |
| No repeat payees | new-beneficiary rate 99.8% for normals vs 100% for anomalies |
| Amount signal inverted | 1.86% of normals had per-customer z>3; **0.00%** of anomalies did |
| Velocity inverted | max 1h txn count: 6 for normals, **1** for anomalies |
| 57% label noise | only 43 of 100 labelled rows differed from the clean baseline at all |
| Useless correlation GT | `bank_cdr_ground_truth.csv` marked 65% of all txns correlated |
| Broken references | 21 of 45 `IPDR_IDs` in the GT pointed to rows not in the IPDR file |
| Too few positives | 100 anomalies / 100,324 txns → ~15–30 in a temporal test split |

**Status: FIXED** by `scripts/generate_dataset.py`.

---

## 2. Current verified state

### 2.1 Dataset (regenerated, 43 MB — was 120 MB)

| File | Rows |
|---|---|
| `data/clean/bank_final.csv` | 20,000 |
| `data/clean/cdr_final.csv` | 60,000 |
| `data/clean/ipdr_final.csv` | 50,000 |
| `data/anomalous/bank_anomaly.csv` | 20,900 |
| `data/anomalous/cdr_anomaly.csv` | 60,653 |
| `data/anomalous/ipdr_anomaly.csv` | 51,433 |
| `data/ground_truth/anomaly_ground_truth.csv` | 800 (3.83% prevalence) |

Shape: 400 customers · 90 days · 15 scenario types · EASY/MEDIUM/HARD tiers.
**CSV column schema is unchanged** — only the statistics changed, so
`src/canonical/*` mappers still apply.

Baseline health (assertions enforced at the end of the generator):

| Metric | Before | After |
|---|---|---|
| odd-hour (0–5h) base rate | 0.249 | 0.039 |
| new-beneficiary base rate | 0.998 | 0.180 |
| amount z>3, normal vs anomaly | 0.0186 / **0.0000** | 0.0108 / **0.2350** |
| bank↔cdr correlation rate | 0.65 | 0.193 |
| cdr↔ipdr correlation rate | 0.79 | 0.057 |
| labelled rows with no injection | 57/100 | 0/800 |

### 2.2 Model results — `python scripts/train.py`

Temporal split 60/15/25: train 12,540 (497 pos) · val 3,135 (113 pos) · **test
5,225 (190 pos)**. Threshold tuned on val, never on test.

| Model | Set | Feats | PR-AUC | ROC-AUC | Precision | Recall | F1 | P@100 |
|---|---|---|---|---|---|---|---|---|
| IsolationForest | A | 25 | 0.116 | 0.667 | 0.134 | 0.147 | 0.140 | 0.18 |
| XGBoost | A (bank) | 25 | 0.411 | 0.741 | 0.686 | 0.311 | 0.428 | 0.64 |
| IsolationForest | B | 42 | 0.229 | 0.852 | 0.244 | 0.268 | 0.256 | 0.24 |
| XGBoost | B (+CDR) | 42 | 0.750 | 0.933 | 0.673 | 0.758 | 0.713 | 0.89 |
| IsolationForest | C | 53 | 0.270 | 0.889 | 0.306 | 0.337 | 0.321 | 0.26 |
| **XGBoost** | **C (+IPDR)** | **53** | **0.829** | **0.972** | **0.799** | **0.774** | **0.786** | **0.90** |
| Stacked_XGB+IF | C | 54 | 0.830 | 0.974 | 0.802 | 0.747 | 0.774 | 0.92 |
| Blend(a=0.95) | C | 53 | 0.797 | 0.975 | 0.799 | 0.774 | 0.786 | 0.89 |
| StackBlend(a=0.95) | C | 54 | 0.827 | 0.975 | 0.802 | 0.747 | 0.774 | 0.92 |

The A→B→C lift (0.41→0.75→0.83) is a real measurement of what fusion adds,
because the correlation links are genuine rather than randomly paired.

**The reported model is selected on validation PR-AUC**, printed every run:

```
XGBoost            val=0.8835  test=0.8285   <- selected
Stacked_XGB+IF     val=0.8600  test=0.8296
Blend(a=0.95)      val=0.8797  test=0.7966
StackBlend(a=0.95) val=0.8574  test=0.8273
```

Note the four are within 0.003 test PR-AUC of each other except the plain Blend.
**The IF-stacking is not a real improvement** — it wins test by 0.001 and loses
val by 0.024. Do not describe it as an architectural breakthrough; if you drop
it, nothing measurable is lost. See section 7 trap 4.

Per-scenario recall @ top-190, Set C — 14 of 15 scenarios detected:

```
1.000  AMOUNT_PLUS_NEW_BENEFICIARY, CALL_THEN_HIGH_VALUE_TRANSFER,
       CALL_THEN_NEW_BENEFICIARY, NETWORK_SESSION_BURST_AROUND_TRANSACTION,
       REPEATED_CALLS_BEFORE_TRANSACTION, SUBTLE_MULTI_SOURCE_SUSPICIOUS_PATTERN
0.923  UNUSUAL_CALL_BEFORE_TRANSACTION
0.857  CUSTOMER_RELATIVE_AMOUNT_SPIKE
0.818  UNUSUAL_LOCATION_CONTEXT
0.786  NEW_DEVICE_AROUND_TRANSACTION
0.750  TRANSACTION_BURST
0.667  IMSI_IMEI_PAIR_NOVELTY
0.500  AMOUNT_VELOCITY_SPIKE     <- see 3.4, data defect
0.500  NEW_BENEFICIARY
0.000  ODD_HOUR_TRANSACTION      <- see 3.1, expected; handled by the rule engine
```

Recall by difficulty: EASY 0.812 · MEDIUM 0.803 · HARD 0.753 — flat, so the
tiers are calibrated rather than collapsing into easy-only detection.

### 2.2b Scoring output — `python scripts/score.py`

Held-out rows only: 184 alerts at risk ≥ 70, **precision 0.799, recall 0.774**.
Bands across all 20,900 rows: CRITICAL 686 · HIGH 111 · MEDIUM 167 · LOW 19,936.
`ODD_HOUR` rule fires on 855 transactions, 62 of them true anomalies.

### 2.3 Files

| Path | Status |
|---|---|
| `scripts/generate_dataset.py` | authoritative dataset builder |
| `scripts/features.py` | **the single feature implementation** — imported by train + score |
| `scripts/train.py` | authoritative training/ablation; writes `models/stage7_setC.joblib` |
| `scripts/score.py` | **NEW** — Stage 8 risk scoring, TreeSHAP reasons, rule engine |
| `models/stage7_setC.joblib` | **NEW** — persisted model + columns + medians + threshold |
| `tests/test_features.py` | 3 tests, passing. **Self-contained — they reimplement the logic rather than importing `features.py`, so they test a copy.** Worth rewiring now that `features.py` is importable. |
| `data/**` | **REGENERATED** |
| `notebook/output/*.csv` | **REGENERATED** from the real data |
| `notebook/Untitled.ipynb` | Cell 2 neutralised to markdown |
| `notebook/stage7.py` | **STALE** — superseded, never ran |
| `notebook/data/` | **STALE** — the 60k-row toy dataset |
| `notebook/Untitled1.ipynb`, `notebook/model_training.ipynb/notebook_1.ipynb` | 1 empty cell each |
| `src/**` (1,772 lines) | **DEAD CODE — nothing imports it.** See 3.2 |
| `tests/` | **Does not exist** despite the README claiming it |

---

## 3. Known open issues

### 3.1 `ODD_HOUR_TRANSACTION` scores 0.000 recall — expected, not a bug

855 transactions occur at night; only 37 are anomalies. A lone odd-hour signal
cannot rank above 190 transactions carrying call/session/device evidence.
Transactions with no telecom context are pushed *below* average, which is why
it is 0.000 rather than the ~0.036 you'd get from random ordering.

This matches the README's "Rules and ML work together" split: **odd-hour alone
belongs in the Rule Engine as a low-severity flag, not in the ML ranker.** Two
valid resolutions — pick one deliberately, do not silently "fix" it:

- **(preferred)** Leave the ML ranker as is; implement odd-hour in the Stage 8
  rule engine. Document the 0.000 as intentional.
- Make it ML-detectable by injecting it only for customers whose own night
  activity is near zero (edit the `ODD_HOUR_TRANSACTION` branch in
  `scripts/generate_dataset.py` to filter anchors by `HOUR_W_CUST[c]`).

### 3.2 `src/` is dead code and duplicates `scripts/train.py`

`grep -rl "from src\|import src"` returns **nothing outside `src/` itself**.
The whole Stage 2–6 pipeline — `canonical/`, `resolution/`, `correlation/`,
`fusion/`, `features/engine.py` — has never been executed. Meanwhile
`scripts/train.py` reimplements the features standalone.

This is the single biggest piece of architectural debt. `src/features/engine.py`
already defines the same features via `FeatureRow`, which `scripts/train.py`
deliberately mirrors by name. **Task 3 in section 6 is to reconcile these.**
Do not add a third implementation.

### 3.3 ~~`AMOUNT_VELOCITY_SPIKE` and `UNUSUAL_LOCATION_CONTEXT`~~ — partly resolved

`UNUSUAL_LOCATION_CONTEXT` 0.545 → **0.818** and `TRANSACTION_BURST` 0.583 →
**0.750** via added features (24h circle-diversity, 2h/6h velocity windows).
`AMOUNT_VELOCITY_SPIKE` stayed at 0.500 — that one is a data defect, see 3.4.

Cost of those features, which must not be forgotten: Set C PR-AUC 0.842 → 0.829,
`NEW_BENEFICIARY` 0.667 → 0.500, `IMSI_IMEI_PAIR_NOVELTY` 0.800 → 0.667, and the
feature count went 40 → 53. Net positive, but it was a trade, not a free win.

### 3.4 CONFIRMED DEFECT — burst clone transactions are labelled `y=0`

`scripts/generate_dataset.py::clone_txn` appends 4–14 sibling transactions for
`AMOUNT_VELOCITY_SPIKE` and `TRANSACTION_BURST`, but **only the anchor is written
to ground truth**. Verified:

```
rows appended by injection: 900
  labelled y=1: 0
  labelled y=0: 900
appended rows span 94 customers; the 108 burst/velocity anchors span the same 94
```

So the model is trained on near-identical feature vectors carrying opposite
labels, which is exactly why it learns to suppress the velocity features. This
caps `AMOUNT_VELOCITY_SPIKE` at 0.500.

**The fix is to label the clones positive, not to exclude them.** A structuring
burst is one suspicious *episode*; in STR reporting every leg is reported.
Excluding them from training was tried and collapses Set C PR-AUC to 0.59,
because it deletes 900 rows and starves the CDR/IPDR scenarios — that result is
evidence the exclusion approach is wrong, not evidence the defect is unfixable.

Implementation: in the `AMOUNT_VELOCITY_SPIKE` / `TRANSACTION_BURST` branch,
collect the IDs returned by `clone_txn` and emit a GT row per clone sharing the
anchor's `Anomaly_ID`, `Scenario_Type` and `Difficulty`. Prevalence rises from
3.83% to roughly 7%; **every number in section 2.2 must be regenerated and this
file updated in the same commit.**

---

## 4. Ground rules for whoever continues

1. **Never regenerate `data/` casually.** It is git-tracked. Run
   `git status data/` first; if dirty, understand why before overwriting.
   Changing `SEED` or any `N_*` constant invalidates every number in section 2 —
   if you do it, re-run training and update this file's tables in the same commit.
2. **Never re-add a data-generation cell to a notebook.** That was root cause
   #1. Notebooks may *read* `data/`; only `scripts/generate_dataset.py` writes it.
3. **Never tune on the test split.** Thresholds come from val. If you add
   hyperparameter search, it searches on val.
4. **Keep features causal.** Every feature must use only events strictly before
   the anchor transaction's timestamp. The helpers `index_by()` / `before()` in
   `scripts/train.py` enforce this — use them rather than writing new lookups.
5. **Do not delete `data/`, `src/`, or notebooks without asking the user.**
   Section 2.3 lists stale files; the user has not yet approved removing them.
6. **Report honestly.** If a change makes a metric worse, say so and keep the
   number. The whole reason this handoff exists is that a 0.52 was reported from
   data that was not the project's data.

---

## 5. Reproducing the current state

```bash
# from the repo root
pip install -r requirements.txt          # pandas, pydantic, pytest
pip install scikit-learn xgboost         # NOT yet in requirements.txt — see Task 0

python scripts/generate_dataset.py       # ~30 s, rewrites data/, self-asserts
python scripts/train.py                  # ~4 min, writes notebook/output/ + models/
python scripts/score.py                  # ~1 min, writes scored_transactions.csv
python -m pytest tests/ -q               # 3 passed
```

Expected tail of `generate_dataset.py`:

```
bank 20,900 | cdr 60,653 | ipdr 51,433 | anomalies 800 (3.83%)
odd-hour(0-5) base rate : 0.039  (was 0.249)
new-beneficiary base    : 0.180  (was 0.998)
amount z>3: base 0.0108 | anom 0.2350  (was 0.0186 / 0.0000)
[OK] injected rows added=900 amount-modified=267
```

If those assertions fail, **stop** — the baseline has lost its structure and no
amount of model tuning will help. That is exactly the failure this whole
exercise was about.

---

## 6. Task list

Ordered by dependency. Each task states its acceptance criterion. Tasks 0–3 are
the ones that matter; 4–7 are the stage roadmap.

### Task 0 — Pin the dependencies · S — **DONE**

`requirements.txt` lists only `pandas, pydantic, pytest`. `scripts/train.py`
needs `scikit-learn` and `xgboost` (installed ad hoc during this work —
xgboost 3.3.0). Add them with version bounds. Add `numpy` explicitly.

**Done when:** a clean venv can run section 5 with no manual `pip install`.

### Task 1 — Close the gap on the three weak scenarios · M — **2 of 3 DONE**

Targets: `AMOUNT_VELOCITY_SPIKE` 0.42, `UNUSUAL_LOCATION_CONTEXT` 0.55,
`TRANSACTION_BURST` 0.58.

Try, in order, measuring after each:

- **Location:** there is currently no feature comparing the CDR circle/BTS at
  transaction time against the customer's *home* circle. `roaming_change` exists
  but only looks at the single most recent call. Add a "distinct circles in
  previous 24h" and "current circle ≠ modal circle" pair over the 24h window
  (the `w24` pattern already used for `imei_novelty`).
- **Velocity/burst:** current windows are 10m/30m/1h. Injected bursts span up to
  58 minutes, so a 2h and 6h window plus "amount sum in window ÷ customer's
  7-day mean" should separate them better.
- Only if features plateau: tune `max_depth`, `min_child_weight`,
  `scale_pos_weight` on **val**.

**Done when:** all three ≥ 0.70 recall @ top-190, with Set C PR-AUC not
regressing below 0.84. Record the before/after per-scenario table.

### Task 2 — Add a runnable check · S — **DONE (see caveat in 2.3)**

There are no tests. Per the repo's working agreement, non-trivial logic leaves
one runnable check behind. Write `tests/test_features.py` (assert-based, no
fixtures) covering the three bugs that actually occurred:

1. **Causality** — build features for a hand-made 5-transaction frame; assert no
   feature value changes when a *later* transaction is appended.
2. **Alignment** — assert `txn_count_previous_1h` for a known customer matches a
   brute-force count. *(This caught the `groupby().rolling()` bug.)*
3. **Dtype** — assert `Sender_Phone_Number` survives CSV round-trip as `str`
   starting with `+91`. *(This caught the int64 coercion bug.)*

**Done when:** `pytest tests/` passes and fails if you revert any of the three
fixes in section 7.

### Task 3 — Reconcile `scripts/train.py` with `src/` · L

See 3.2. Decide **with the user** between:

- **(A)** Wire `src/` up: `loader` → `registry` → `correlation` → `fusion` →
  `features/engine.py`, and have `scripts/train.py` consume `FeatureRow` objects
  instead of computing features itself. Honours the documented Stage 2–6
  architecture; more work; `FeatureEngine` is O(n) per row and may be slow at
  20k×110k events — benchmark before committing.
- **(B)** Delete the unused `src/` modules and treat `scripts/train.py` as the
  pipeline. Much smaller repo; abandons the staged architecture the README and
  `docs/` describe.

**Do not start this without a decision** — it is a large, hard-to-reverse change
either way. Feature names in `scripts/train.py` already mirror
`src/features/models.py::FeatureRow` specifically to make (A) tractable.

**Done when:** exactly one feature implementation exists in the repo, and
Set C PR-AUC is within ±0.02 of 0.842.

### Task 4 — Stage 8: risk scoring, explainability, rule engine · M — **CORE DONE**

Roadmap stage 8. Convert the XGBoost probability into a 0–100 risk score with
per-transaction reason codes (SHAP or the ranked feature contributions).
Implement the rule engine here, including odd-hour per 3.1.

`scripts/score.py` delivers this: 0-100 risk score (threshold pinned to 70),
band, exact TreeSHAP top-3 reasons, and the `ODD_HOUR` rule. Remaining: expand
the rule set beyond the single documented rule, and wire scoring into the API.

**Done when:** every flagged transaction carries its top-3 contributing features
and any fired rules.

### Task 5 — Stage 9: graph analytics · M

Build the sender→receiver / subscriber→contact graph. The generator creates
60 mule accounts that receive only anomalous funds — they should surface as
high-in-degree, low-out-degree nodes. Good validation signal.

### Task 6 — Stages 10–12: API, dashboard, STR reporting · L

`backend/` is a FastAPI app currently serving only the PDF parser
(`backend/pdf/`). Extend it with investigation search endpoints over the scored
transactions, then the dashboard and forensic report export.

### Task 7 — Housekeeping · S

Needs user approval (see ground rule 5). Candidates: `notebook/stage7.py`,
`notebook/data/`, `notebook/Untitled1.ipynb`,
`notebook/model_training.ipynb/notebook_1.ipynb` (a *directory* named
`.ipynb`, which is confusing). Rename `notebook/Untitled.ipynb` to something
meaningful. Update the README: it claims a `tests/` dir and a `notebooks/`
dir (actual name is `notebook/`), and the stage table still says Stage 1
COMPLETED / Stage 2 NEXT.

---

## 7. Traps — read before editing `scripts/train.py`

These three bugs were hit during this work. All are fixed; each is easy to
reintroduce.

1. **`groupby().rolling()` returns rows in group order, not row order.**
   Assigning `.values` back to a column silently scrambles every value. It made
   `TRANSACTION_BURST` recall 0.125 instead of 0.583. `scripts/train.py` now
   uses the explicit positional `index_by()`/`before()` helpers instead. Do not
   reintroduce `groupby().rolling()` without reindexing on the original index.

2. **Pandas parses `+919812345678` as int64.** This silently breaks every phone
   join between bank/CDR/IPDR — no error, just zero matches and dead telecom
   features. All join keys are read via the `STR` dtype dict at the top of
   `scripts/train.py`. Add any new key column to it.

4. **Selecting the reported model on the test set.** `train.py` briefly chose its
   Set C model with `max(c_models, key=lambda x: average_precision_score(y[te], x[0]))`.
   That leaks the held-out set into model choice: it picked Stacked_XGB+IF, which
   wins test by 0.001 and loses val by 0.024, and inflated the per-scenario table
   (`TRANSACTION_BURST` read 0.833 instead of 0.750). Selection now uses the val
   PR-AUCs, which were already being computed. **Ground rule 3 exists because of
   this — the whole point of the split is that test is touched once.**

3. **Per-hour bins cannot be estimated from ~52 transactions per customer.**
   `hour_rarity` was near-constant until it became a Laplace-smoothed 6-hour
   bucket. Watch for the same trap in any new per-customer categorical rate.

Also: `np.char.add` requires matching dtype kinds — `pd.Series.dt.strftime(...)`
returns `object` and must be cast (`.astype("<U6")`) before use. This bit twice
in `generate_dataset.py`.

---

## 8. Quick reference

```
scripts/generate_dataset.py   dataset builder      SEED=42, self-asserting
scripts/features.py           THE feature builder  imported by train.py + score.py
scripts/train.py              ablation + tuning    Sets A/B/C, val-selected, persists model
scripts/score.py              risk scores          TreeSHAP reasons + rule engine
models/stage7_setC.joblib     trained bundle       model + columns + medians + threshold
data/clean/                   baseline, no anomalies
data/anomalous/               baseline + 800 injected anomalies  <- train on this
data/ground_truth/            labels + correlation GT
notebook/output/              stage7_ablation_results.csv, hackathon_results.csv
src/                          Stage 2-6 canonical pipeline (currently unused)
backend/                      FastAPI, PDF parser only
docs/TRI_NETRA_STAGE_WISE_DOCUMENTATION.md   stage specs
```

**Anchor principle from the README, still true:** one model observation = one
bank transaction. CDR and IPDR supply context around it; they are never
observations themselves.
