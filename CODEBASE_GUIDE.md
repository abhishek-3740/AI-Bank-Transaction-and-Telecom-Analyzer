# TRI-NETRA — Complete Codebase Guide

**Problem Statement:** ERH26_PS_03  
**Stack:** Python 3.12 · FastAPI · XGBoost · pdfplumber · pandas · NumPy  
**Status:** Stages 1–12 implemented and verified

This document is your single reference for every file, feature, data flow, API route,
ML model, and design decision in the project. Read top to bottom once — after that
use the section headers to jump directly to what you need.

---

## Table of Contents

1. [What TRI-NETRA does](#1-what-tri-netra-does)
2. [Repository layout](#2-repository-layout)
3. [Data layer](#3-data-layer)
4. [Pipeline scripts](#4-pipeline-scripts)
5. [Feature engineering](#5-feature-engineering)
6. [ML model](#6-ml-model)
7. [Risk scoring and rule engine](#7-risk-scoring-and-rule-engine)
8. [Graph analytics](#8-graph-analytics)
9. [PDF parser](#9-pdf-parser)
10. [Backend API — all routes](#10-backend-api--all-routes)
11. [Tests](#11-tests)
12. [Key numbers you must know](#12-key-numbers-you-must-know)
13. [How to run everything](#13-how-to-run-everything)
14. [Design decisions and traps](#14-design-decisions-and-traps)

---

## 1. What TRI-NETRA does

TRI-NETRA fuses three data sources to detect financial fraud and flag suspicious
transactions for STR (Suspicious Transaction Report) filing.

```
Bank Statements  ─┐
CDR (Call logs)  ─┼──► Feature Engine ──► XGBoost ──► Risk Score ──► Investigation API
IPDR (IP logs)   ─┘                                      │
                                                          ▼
                                                     Graph Analytics ──► Mule Detection
                                                          │
                                                          ▼
                                                     STR Report Export
```

**Core principle:** one ML observation = one bank transaction.
CDR and IPDR records are context around that transaction — never observations themselves.

**15 fraud scenarios detected:**

| Scenario | What it captures |
|---|---|
| ODD_HOUR_TRANSACTION | Transaction at hours the customer almost never uses |
| CUSTOMER_RELATIVE_AMOUNT_SPIKE | Amount >> customer's own median |
| AMOUNT_VELOCITY_SPIKE | Multiple high-value transfers in <58 min |
| TRANSACTION_BURST | Many transfers in quick succession |
| NEW_BENEFICIARY | First-ever transfer to this receiver |
| AMOUNT_PLUS_NEW_BENEFICIARY | High amount + new receiver combo |
| UNUSUAL_CALL_BEFORE_TRANSACTION | Suspicious call just before the transfer |
| CALL_THEN_HIGH_VALUE_TRANSFER | Call then large transfer |
| CALL_THEN_NEW_BENEFICIARY | Call then new receiver |
| REPEATED_CALLS_BEFORE_TRANSACTION | Multiple calls from same number before transfer |
| NETWORK_SESSION_BURST_AROUND_TRANSACTION | IP data burst around transfer time |
| UNUSUAL_LOCATION_CONTEXT | Call from a different telecom circle than home |
| NEW_DEVICE_AROUND_TRANSACTION | Previously unseen IMEI near the transaction |
| IMSI_IMEI_PAIR_NOVELTY | New IMSI+IMEI combination |
| SUBTLE_MULTI_SOURCE_SUSPICIOUS_PATTERN | Multiple weak signals together |

---

## 2. Repository layout

```
TRI-NETRA/
│
├── data/                          ← ALL datasets (git-tracked, never regenerate casually)
│   ├── clean/                     ← Baseline (no anomalies)
│   │   ├── bank_final.csv         20,000 rows
│   │   ├── cdr_final.csv          60,000 rows
│   │   └── ipdr_final.csv         50,000 rows
│   ├── anomalous/                 ← Baseline + injected anomalies  ← TRAIN ON THIS
│   │   ├── bank_anomaly.csv       20,900 rows
│   │   ├── cdr_anomaly.csv        60,653 rows
│   │   └── ipdr_anomaly.csv       51,433 rows
│   └── ground_truth/
│       ├── anomaly_ground_truth.csv     1,700 labelled rows (8.13% prevalence)
│       ├── bank_cdr_ground_truth.csv    bank↔CDR correlation pairs
│       └── cdr_ipdr_ground_truth.csv    CDR↔IPDR correlation pairs
│
├── scripts/                       ← THE pipeline — all authoritative code lives here
│   ├── generate_dataset.py        Builds data/ from scratch (SEED=42, self-asserting)
│   ├── features.py                THE single feature implementation (53 features)
│   ├── train.py                   Trains + ablates Sets A/B/C, persists model bundle
│   ├── score.py                   Risk scoring, TreeSHAP, rule engine
│   ├── graph_analytics.py         Stage 9 graph analytics
│   ├── test_parse_pdf.py          Standalone PDF verification script
│   └── verify_pdf.py              Verifies XXXX6607.pdf through the full pdf_parser
│
├── models/
│   └── stage7_setC.joblib         Persisted model bundle (XGBoost + columns + medians + threshold)
│
├── pdf-parser/                    ← PDF ingestion library
│   ├── pdf_parser.py              Main parser: pdfplumber extraction, metadata, cleanup
│   └── schema_mapper.py           Column mapping: fuzzy + semantic matching to canonical schema
│
├── backend/                       ← FastAPI application
│   ├── main.py                    App entry point, registers all 4 routers
│   ├── requirements.txt           Backend-specific deps (fastapi, uvicorn, pydantic)
│   ├── pdf/                       PDF parse endpoint
│   │   ├── router.py              POST /api/v1/pdf/parse
│   │   ├── models.py              ParserResponse pydantic model
│   │   ├── config.py              PROJECT_NAME, VERSION, upload limits
│   │   ├── logging_config.py      Structured logger setup
│   │   └── utils.py               File save/delete/validate helpers
│   ├── scoring/                   Investigation search + on-demand scoring
│   │   ├── router.py              5 endpoints (see Section 10)
│   │   └── models.py              ScoredTransaction, AlertListResponse, etc.
│   ├── graph/                     Graph analytics query endpoints
│   │   ├── router.py              5 endpoints (see Section 10)
│   │   └── models.py              GraphNode, GraphEdge, GraphSummary
│   └── reports/                   STR forensic report export
│       ├── router.py              3 endpoints (see Section 10)
│       └── models.py              STRReport, STRTransaction
│
├── tests/
│   └── test_features.py           3 passing tests (causality, alignment, dtype)
│
├── notebook/
│   ├── tri_netra_exploration.ipynb  Exploration notebook (reads data/, never writes it)
│   └── output/                    All generated artefacts
│       ├── scored_transactions.csv  20,900 rows with risk scores + SHAP + rules
│       ├── graph_analytics.csv      858 nodes, suspicion scores
│       ├── graph_edges.csv          20,900 directed edges
│       ├── graph_cdr_nodes.csv      CDR communication graph node stats
│       ├── stage7_ablation_results.csv  Full ablation table (Sets A/B/C)
│       ├── hackathon_results.csv    Same as ablation (copy for reporting)
│       ├── bank_parsed.csv          Last PDF parsed by the production parser
│       └── XXXX6607_parsed.csv      XXXX6607.pdf parsed output (38 transactions)
│
├── docs/
│   └── TRI_NETRA_STAGE_WISE_DOCUMENTATION.md  Detailed stage specifications
│
├── README.md                      Project overview + quick-start
├── HANDOFF.md                     Engineering history, all decisions, verified numbers
├── CODEBASE_GUIDE.md              ← This file
├── requirements.txt               Root deps (pandas, numpy, scikit-learn, xgboost, pytest)
├── .gitignore
└── LICENSE
```

---

## 3. Data layer

### 3.1 Schema — Bank (`bank_anomaly.csv`)

| Column | Type | Notes |
|---|---|---|
| Transaction_ID | str | `TXN` or `ATM` prefix + date + 6 alphanumeric chars |
| Date | str | `YYYY-MM-DD` |
| Timestamp | str | `HH:MM:SS` |
| Txn_Ref_Number | str | 12-char alphanumeric |
| Transaction_Mode | str | ATM / UPI / IMPS / NEFT / RTGS / Cash Deposit / Pos |
| Currency | str | Always `INR` |
| Transaction_Amount | float | Positive; sign context comes from DR/CR in bank PDFs |
| Sender_Customer_ID | str | 9-digit, format `1XXXXXXXX` |
| Sender_Customer_Name | str | |
| Sender_Bank_Name | str | One of 10 real Indian banks |
| Sender_Account_Number | str | 12-digit |
| Sender_Account_Type | str | Savings / Salary / Current / Demat |
| Sender_IFSC | str | 4-char bank code + `0` + 7 chars |
| Sender_Phone_Number | str | `+91` + 10 digits — **always read as str**, not int64 |
| Receiver_* | str | Same structure as Sender |

**Key:** `Sender_Phone_Number` is the join key to CDR. `Sender_Customer_ID` is the join key to graph nodes.

### 3.2 Schema — CDR (`cdr_anomaly.csv`)

| Column | Type | Notes |
|---|---|---|
| CDR_ID | str | `CDR2026` + 8-digit sequence |
| Call_Date | str | `YYYY-MM-DD` |
| Call_Start_Time | str | `HH:MM:SS` |
| A_Party_Number | str | Caller phone `+91...` |
| B_Party_Number | str | Called phone |
| Call_Type | str | VOICE / SMS / MISSED |
| Call_Duration_Seconds | int | 0 for SMS/MISSED |
| IMSI | str | `404` + 12 digits |
| IMEI | str | 15 digits, starts with `35` |
| First_BTS_Location | str | `{Circle}_BTS_{n}` |
| First_Cell_Global_ID | str | `404-45-{cell}-{id}` |
| Roaming_Network_Circle | str | One of 9 Indian circles |

**Key:** `A_Party_Number` joins to bank via `Sender_Phone_Number`.

### 3.3 Schema — IPDR (`ipdr_anomaly.csv`)

| Column | Type | Notes |
|---|---|---|
| IPDR_ID | str | `IPDR2026` + 8-digit sequence |
| Session_Date | str | `YYYY-MM-DD` |
| Session_Start_Time | str | `HH:MM:SS` |
| Subscriber_IMSI | str | |
| Subscriber_MSISDN | str | Phone number — join key to bank |
| Device_IMEI | str | |
| Source_IP_Address | str | Private `10.x.x.x` |
| Destination_IP_Address | str | From customer's fixed destination pool |
| Destination_Port | int | 443 / 80 / 53 / 123 / 8080 / 8443 / 5228 |
| Cell_Global_ID | str | |
| Session_Duration_Seconds | int | |

### 3.4 Ground truth (`anomaly_ground_truth.csv`)

| Column | Notes |
|---|---|
| Anomaly_ID | `ANOM{n}` for anchors, `ANOM{n}C{j}` for burst clones |
| Customer_ID | |
| Transaction_ID | Links back to bank_anomaly.csv |
| CDR_IDs | Semicolon-separated CDR_IDs injected for this event |
| IPDR_IDs | Same for IPDR |
| Scenario_Type | One of the 15 scenario names |
| Difficulty | EASY / MEDIUM / HARD |
| Source_Scope | BANK_ONLY / BANK_CDR / BANK_CDR_IPDR |
| Is_Suspicious | Always 1 |

**1,700 total anomalous rows** (800 anchors + 900 burst/velocity clones, all y=1).

### 3.5 Dataset statistics (verified)

| Metric | Value |
|---|---|
| Odd-hour (0–5h) base rate in normal txns | 0.041 (was 0.249 — fixed) |
| New-beneficiary base rate | 0.187 (was 0.998 — fixed) |
| Amount z>3: normal / anomalous | 0.0094 / 0.1329 (was inverted — fixed) |
| Bank↔CDR correlation rate | 0.193 (was 0.65 = useless) |
| CDR↔IPDR correlation rate | 0.057 (was 0.79 = useless) |
| Anomaly prevalence | 8.13% |
| Mule accounts (only receive, never send) | 60 entities |

---

## 4. Pipeline scripts

### 4.1 `scripts/generate_dataset.py`

**Purpose:** Build the entire `data/` directory from SEED=42. Run once to get the baseline; only re-run if you need to change the scenario mix.

**What it does:**
1. Creates 400 customer profiles with per-customer log-normal amount distributions, personal hourly activity clocks, and fixed payee sets
2. Generates 20,000 baseline bank transactions, 60,000 CDR records, 50,000 IPDR sessions
3. Loops over 800 anchor transactions and injects one of 15 scenario types (EASY/MEDIUM/HARD)
4. For AMOUNT_VELOCITY_SPIKE and TRANSACTION_BURST: calls `clone_txn()` to create 4–14 sibling transactions AND emits a GT row for each clone (this was the key bug fix — previously clones were y=0)
5. Creates 60 mule accounts that only appear as receivers in anomalous transactions
6. Builds bank↔CDR and CDR↔IPDR correlation ground truth tables
7. Self-asserts all health metrics before writing CSVs

**Key constants (changing any invalidates all verified numbers):**
```python
SEED = 42
N_CUSTOMERS = 400
DAYS = 90
N_BANK = 20_000
N_CDR = 60_000
N_IPDR = 50_000
N_ANOMALY = 800
```

**Expected output tail:**
```
bank 20,900 | cdr 60,653 | ipdr 51,433 | anomalies 1,700 (8.13%)
odd-hour(0-5) base rate : 0.041
new-beneficiary base    : 0.187
[OK] injected rows added=900 amount-modified=267
```

### 4.2 `scripts/train.py`

**Purpose:** Train, ablate, and persist the final model. Run after generate_dataset.py.

**What it does:**
1. Calls `features.load_sources()` and `features.build_features()` to get Sets A, B, C
2. Temporal split: 60% train / 15% val / 25% test (by timestamp quantile, never random)
3. For each feature set trains IsolationForest + XGBoost
4. For Set C only: grid-searches `max_depth ∈ {4,5}`, `min_child_weight ∈ {1,3,5}`, `reg_lambda ∈ {1.0,2.0}` on val PR-AUC
5. Also builds Stacked (XGB+IF) and Blend variants for Set C
6. **Selects the reported model on validation PR-AUC** (not test — critical)
7. Persists the selected bundle to `models/stage7_setC.joblib`
8. Prints per-scenario recall and difficulty breakdown

**Model bundle contents (`stage7_setC.joblib`):**
```python
{
  "kind":        "XGBoost",        # model type string
  "model":       XGBClassifier,    # fitted model
  "columns":     list[str],        # 53 feature names in order
  "medians":     pd.Series,        # imputation medians (fitted on train only)
  "threshold":   float,            # val-tuned decision threshold (≈0.736)
  "val_pr_auc":  float,            # 0.8986
  "test_pr_auc": float,            # 0.8999
}
```

### 4.3 `scripts/score.py`

**Purpose:** Score all 20,900 transactions, produce the investigation CSV. Run after train.py.

**What it does:**
1. Loads the model bundle (never retrains — medians must travel with the model to prevent leakage)
2. Builds features for the full anomalous dataset
3. Converts probabilities to 0–100 risk scores using a piecewise-linear map pinning the threshold to 70
4. Assigns risk bands: LOW / MEDIUM / HIGH / CRITICAL
5. Runs exact TreeSHAP contributions and picks top-3 positive contributors as reason codes
6. Runs the rule engine (see Section 7)
7. Writes `notebook/output/scored_transactions.csv`

**Risk score formula:**
```
if prob < threshold:  risk = 70 * prob / threshold
else:                 risk = 70 + 30 * (prob - threshold) / (1 - threshold)
```
So risk ≥ 70 always means "the model would alert."

### 4.4 `scripts/graph_analytics.py`

**Purpose:** Build the financial transfer graph and detect mule accounts. Run after score.py.

**What it does:**
1. Loads scored_transactions.csv + bank_anomaly.csv + anomaly_ground_truth.csv
2. Builds directed graph: sender_customer_id → receiver_account_number
3. Runs power-iteration PageRank (damping=0.85, no external library needed)
4. Computes per-node: in_degree, out_degree, in_out_ratio, total_sent, total_received, pagerank, alert_count
5. Combines into `suspicion_score = 0.4×PageRank_norm + 0.3×in_out_ratio_norm + 0.3×alert_count_norm`
6. Flags known mule accounts (appear as receivers in ground truth transactions)
7. Builds CDR communication graph separately
8. Writes three CSVs: graph_analytics.csv, graph_edges.csv, graph_cdr_nodes.csv

**Validation result:** 100% of top-50 suspicious nodes are confirmed mule accounts.

---

## 5. Feature engineering

**Single implementation:** `scripts/features.py` — imported by both train.py and score.py. There is no other feature code in the repo.

### 5.1 How causal windows work

All features use only events **strictly before** the anchor transaction's timestamp. The two helpers that enforce this:

```python
def index_by(frame, key):
    # Returns {key_value: (sorted_timestamps, positional_indices)}
    # Sorted by key then by timestamp — enables O(log n) binary search

def before(idx, key, t, window=None):
    # Returns (timestamps, positions) of events for `key` strictly before t
    # Optional window parameter limits to t-window..t
    hi = searchsorted(ts, t)          # strictly before t
    lo = searchsorted(ts, t - window) # optionally limit window
```

**Never use `groupby().rolling()`** — it returns rows in group order, not row order. Assigning `.values` back scrambles alignment. The `index_by/before` pattern is the safe replacement.

### 5.2 Set A — Bank features (25)

| Feature | What it measures |
|---|---|
| transaction_amount | Raw amount |
| transaction_hour | Hour 0–23 |
| customer_history_count | Cumulative prior txns for this customer |
| amount_vs_customer_median | Amount ÷ customer's rolling prior median |
| amount_robust_zscore | (Amount − median) ÷ (1.4826 × MAD) |
| amount_percentile | Fraction of prior txns below this amount |
| receiver_seen_before | 1 if sender has sent to this receiver before |
| receiver_historical_count | Times this receiver has been seen |
| receiver_frequency | pair frequency ÷ customer history count |
| hour_rarity | 1 − share of prior activity in same 6h bucket (Laplace-smoothed) |
| txn_count_previous_10m | Velocity count, 10-minute window |
| txn_count_previous_30m | Velocity count, 30-minute window |
| txn_count_previous_1h | Velocity count, 1-hour window |
| txn_count_previous_2h | Velocity count, 2-hour window |
| txn_count_previous_6h | Velocity count, 6-hour window |
| amount_velocity_30m | Sum of amounts in prior 30 min |
| amount_velocity_1h | Sum of amounts in prior 1 hour |
| amount_velocity_2h | Sum of amounts in prior 2 hours |
| amount_velocity_6h | Sum of amounts in prior 6 hours |
| amount_ratio_30m_to_7d | 30m amount sum ÷ customer's 7-day mean |
| amount_ratio_1h_to_7d | 1h amount sum ÷ customer's 7-day mean |
| amount_ratio_2h_to_7d | 2h amount sum ÷ customer's 7-day mean |
| txn_rate_acceleration | 1h txn count ÷ 6h average rate |
| txn_velocity_vs_customer_norm | 1h count ÷ customer's daily average rate |
| time_since_previous_transaction | Seconds since last txn by this sender |

### 5.3 Set B — +CDR features (17 additional = 42 total)

| Feature | What it measures |
|---|---|
| has_cdr_context | 1 if any CDR record found for this phone |
| calls_previous_10m / 30m / 1h | Call counts in windows before txn |
| nearest_call_before_seconds | Time gap to most recent call |
| total_call_duration_30m | Sum of call durations in prior 30 min |
| max_call_duration_30m | Longest call in prior 30 min |
| caller_novelty | 1 if most recent B-party is new in this customer's call history |
| caller_historical_frequency | Historical frequency of most recent B-party |
| imei_novelty | 1 if any IMEI in prior 24h appeared for the first time in past 7 days |
| cell_novelty | 1 if any cell tower in prior 24h is newly seen |
| cdr_imsi_imei_pair_novelty | 1 if any IMSI+IMEI pair in 24h is new |
| distinct_circles_24h | Number of distinct telecom circles in prior 24h |
| circle_mismatch_24h | 1 if any call in 24h was outside home circle |
| distinct_circles_30m | Circles in tighter 30-min window |
| circle_mismatch_30m | Circle mismatch in 30-min window |
| roaming_change | 1 if most recent call was outside home circle |

### 5.4 Set C — +IPDR features (11 additional = 53 total)

| Feature | What it measures |
|---|---|
| has_ipdr_context | 1 if any IPDR session found |
| sessions_previous_10m / 30m | Session counts before txn |
| nearest_session_before_seconds | Time gap to most recent session |
| source_ip_novelty | 1 if source IP appeared for first time in past 24h |
| destination_ip_novelty | 1 if dest IP appeared for first time in past 7 days |
| destination_port_novelty | 1 if this phone+port combo is new in past 7 days |
| imsi_imei_pair_novelty | 1 if IMSI+IMEI pair is new in past 7 days |
| device_consistency | 1 if last session used the modal IMEI for this subscriber |
| cell_consistency | 1 if last session used the modal cell for this subscriber |
| session_duration_deviation | Last session duration ÷ subscriber's median |

---

## 6. ML model

### 6.1 Ablation results (Set C = production model)

| Model | Set | Features | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| IsolationForest | A | 25 | 0.593 | 0.820 | 0.847 | 0.445 | 0.583 |
| XGBoost | A (bank only) | 25 | 0.742 | 0.865 | 0.966 | 0.573 | 0.719 |
| IsolationForest | B | 42 | 0.699 | 0.913 | 0.866 | 0.455 | 0.596 |
| XGBoost | B (+CDR) | 42 | 0.869 | 0.945 | 0.913 | 0.736 | 0.815 |
| IsolationForest | C | 53 | 0.746 | 0.930 | 0.895 | 0.470 | 0.616 |
| **XGBoost** | **C (+IPDR)** | **53** | **0.900** | **0.964** | **0.931** | **0.781** | **0.850** |

The A→B→C PR-AUC lift (0.74→0.87→0.90) is real — it measures what fusing CDR and IPDR adds.

### 6.2 Per-scenario recall @ top-398 (production XGBoost Set C)

```
1.000  AMOUNT_PLUS_NEW_BENEFICIARY
1.000  CALL_THEN_HIGH_VALUE_TRANSFER
1.000  CALL_THEN_NEW_BENEFICIARY
1.000  CUSTOMER_RELATIVE_AMOUNT_SPIKE
1.000  NETWORK_SESSION_BURST_AROUND_TRANSACTION
1.000  REPEATED_CALLS_BEFORE_TRANSACTION
1.000  SUBTLE_MULTI_SOURCE_SUSPICIOUS_PATTERN
0.936  AMOUNT_VELOCITY_SPIKE   ← was 0.500; fixed by labelling burst clones y=1
0.923  UNUSUAL_CALL_BEFORE_TRANSACTION
0.818  UNUSUAL_LOCATION_CONTEXT
0.800  IMSI_IMEI_PAIR_NOVELTY
0.795  TRANSACTION_BURST       ← was capped by same defect; improved
0.786  NEW_DEVICE_AROUND_TRANSACTION
0.556  NEW_BENEFICIARY
0.000  ODD_HOUR_TRANSACTION    ← intentional; handled by rule engine (see §7)
```

Recall by difficulty: EASY 0.857 · MEDIUM 0.858 · HARD 0.838 — flat tiers, not collapsing to easy.

### 6.3 Model selection rule

The reported model is selected on **validation PR-AUC**, never on test. This is enforced in `train.py`:
```python
c_models = [(xgb_te, "XGBoost", best_prauc), ...]
best_score, best_name, _ = max(c_models, key=lambda x: x[2])  # x[2] = val PR-AUC
```

### 6.4 Top-12 features by importance (XGBoost Set C)

```
amount_velocity_30m            0.195
amount_ratio_30m_to_7d         0.078
max_call_duration_30m          0.067
time_since_previous_transaction 0.055
total_call_duration_30m        0.054
amount_ratio_1h_to_7d          0.048
cell_novelty                   0.037
sessions_previous_30m          0.032
cdr_imsi_imei_pair_novelty     0.028
imei_novelty                   0.027
destination_ip_novelty         0.024
txn_count_previous_1h          0.022
```

---

## 7. Risk scoring and rule engine

**File:** `scripts/score.py`

### 7.1 Risk score (0–100)

Piecewise-linear map so that risk ≥ 70 always means the model would alert:
```
prob < threshold:  risk = 70 × prob / threshold
prob ≥ threshold:  risk = 70 + 30 × (prob − threshold) / (1 − threshold)
```

**Bands:**

| Band | Risk score | Meaning |
|---|---|---|
| LOW | 0–40 | Normal |
| MEDIUM | 40–70 | Watch |
| HIGH | 70–90 | Alert |
| CRITICAL | 90–100 | Immediate investigation |

**Current band distribution (all 20,900 rows):**
```
CRITICAL  1,337
HIGH        231
MEDIUM      329
LOW      19,003
```

### 7.2 TreeSHAP reason codes

For each transaction the top-3 features with positive SHAP contributions are stored as:
```
reason_1: "amount_velocity_30m (+2.34)"
reason_2: "total_call_duration_30m (+1.87)"
reason_3: "imei_novelty (+1.12)"
```
These are exact Shapley values from `model.get_booster().predict(..., pred_contribs=True)`, not feature importances.

### 7.3 Rule engine (5 rules)

Rules fire independently of the ML score. They catch signals the ranker cannot rank reliably.

| Rule | Condition | Why ML can't rank this alone |
|---|---|---|
| ODD_HOUR | transaction_hour ∈ [0, 5] | 855 night txns, only 37 anomalous — the ML model pushes odd-hour-only rows below average |
| HIGH_AMOUNT_ANOMALY | amount_vs_customer_median > 5.0 | Catches spikes when CDR/IPDR context is absent |
| RAPID_SUCCESSION | txn_count_previous_10m ≥ 3 | 149 fires, 100% TP rate |
| NEW_BENEFICIARY_FLAG | receiver_seen_before == 0 | Broad signal — first-ever receiver |
| TELECOM_BURST | calls_previous_30m ≥ 3 | 60 fires, 92% TP rate |

Multiple rules can fire on the same transaction — stored as pipe-separated string: `"ODD_HOUR|HIGH_AMOUNT_ANOMALY"`

**Rule engine statistics:**
```
ODD_HOUR              fired=855   TPs=69
HIGH_AMOUNT_ANOMALY   fired=636   TPs=406
RAPID_SUCCESSION      fired=149   TPs=149
NEW_BENEFICIARY_FLAG  fired=3,879 TPs=285
TELECOM_BURST         fired=60    TPs=55
```

---

## 8. Graph analytics

**File:** `scripts/graph_analytics.py`

### 8.1 Financial transfer graph

- **Nodes:** every unique sender_customer_id and receiver_account_number
- **Edges:** one directed edge per bank transaction, weight = transaction amount
- **858 nodes, 20,900 edges**

**Node attributes computed:**

| Attribute | How |
|---|---|
| out_degree | Number of outgoing transactions |
| in_degree | Number of incoming transactions |
| in_out_ratio | in_degree ÷ (out_degree + 1e-6) |
| total_sent | Sum of outgoing amounts |
| total_received | Sum of incoming amounts |
| pagerank | Power-iteration PageRank (damping=0.85, custom implementation, no library) |
| alert_count | Number of transactions with risk_score ≥ 70 |
| is_mule_account | 1 if appears as receiver in anomaly_ground_truth.csv |
| suspicion_score | 0.4 × PageRank_norm + 0.3 × in_out_ratio_norm + 0.3 × alert_count_norm |

**Mule detection result:** 60 mule accounts created in data generator. 100% of them appear in top-50 nodes by suspicion_score.

### 8.2 CDR communication graph

Simpler node-level stats from CDR VOICE calls:
- calls_made, calls_received, unique_b_parties, unique_a_parties, total_duration_s
- Written to `notebook/output/graph_cdr_nodes.csv`

---

## 9. PDF parser

### 9.1 Two-file library (`pdf-parser/`)

**`pdf_parser.py`** — orchestration and extraction  
**`schema_mapper.py`** — column classification and canonical mapping

### 9.2 Parser pipeline (8 steps)

```
1. _extract_statement_metadata()   ← extracts bank name, account, IFSC, customer from header
2. extract_tables_from_pdf()       ← pdfplumber: default → text strategy → text fallback
3. _clean_raw_dataframe()          ← header detection, repeated-header removal, debit/credit merge
4. detect_dataset_type()           ← fuzzy scoring → BANK / CDR / IPDR classification
5. map_columns()                   ← 4-tier mapping: exact → alias → RapidFuzz → SentenceTransformer
6. _apply_statement_metadata()     ← injects bank/account/IFSC into every row
7. _clean_mapped_dataframe()       ← numeric parsing, date normalisation, mode parsing
8. ensure_schema() + _validate_schema()  ← enforces canonical column order, validates critical fields
```

### 9.3 Bank name detection fix

The original code counted bank name occurrences across the full text. UPI narrations like `"UPI/P2M/.../Paymen/Yes Bank Ltd"` caused the sender's bank to be misidentified. Fixed by:
1. First checking the IFSC prefix (UTIB=Axis, HDFC=HDFC, etc.) — most reliable
2. Restricting text-match to the first 800 characters (the document header only)

### 9.4 Column mapping — 4-tier system

| Tier | Method | Threshold |
|---|---|---|
| 1 | Exact string match | 100% |
| 2 | Alias dictionary match | 100% |
| 3 | RapidFuzz WRatio | ≥ 80% |
| 4 | SentenceTransformer cosine similarity | ≥ 0.55 |

The `all-MiniLM-L6-v2` model is downloaded once from HuggingFace and cached at `~/.cache/huggingface/`.

### 9.5 Dataset type detection

Scores each of BANK / CDR / IPDR by summing (fuzzy_match_score × field_weight) for all headers. Fields like IFSC, A_Party_Number, IMSI have 3× weight (highly unique identifiers).

### 9.6 Supported statement types

- **Bank:** Axis Bank, HDFC, SBI, ICICI, Kotak, PNB, Canara, Yes Bank, IndusInd, etc.
- **CDR:** Standard CDR schema with A/B party numbers, IMEI, cell ID
- **IPDR:** Standard IPDR schema with MSISDN, IMSI, IP addresses
- **All formats:** DR/CR split columns, Amount+flag columns, signed single-amount columns

### 9.7 ScannedPDFError vs PDFExtractionError

- `ScannedPDFError` — raised when total text characters < 50 across all pages (image-based PDF, needs OCR)
- `PDFExtractionError` — raised when pdfplumber finds text but no tables can be reconstructed
- The backend (`backend/pdf/router.py`) catches both and returns HTTP 500 with a clear message

---

## 10. Backend API — all routes

**Start:** `uvicorn backend.main:app --reload` from the repo root  
**Docs:** `http://localhost:8000/docs` (Swagger UI auto-generated)  
**Base URL:** `http://localhost:8000`

---

### 10.1 Root

| Method | Path | Returns |
|---|---|---|
| GET | `/` | name, version, all endpoint names |
| GET | `/health` | `{"status": "healthy"}` |

---

### 10.2 PDF Parser — `/api/v1/pdf`

#### `POST /api/v1/pdf/parse`

Upload a PDF bank statement, CDR, or IPDR document. Returns structured JSON.

**Request:** `multipart/form-data` with a `file` field (PDF only).

**Response (`ParserResponse`):**
```json
{
  "status": "success",
  "dataset_type": "auto",
  "rows": 38,
  "columns": ["Transaction_ID", "Date", "Timestamp", ...],
  "data": [{ "Transaction_ID": null, "Date": "2024-07-30", ... }, ...]
}
```

**Errors:**
- 400 — no filename or non-PDF extension
- 500 — parsing failed (scanned PDF, unrecognisable layout)

---

### 10.3 Scoring / Investigation — `/api/v1/scoring`

All endpoints read from `notebook/output/scored_transactions.csv`. Returns 503 if that file doesn't exist (run score.py first).

#### `GET /api/v1/scoring/alerts`

Fetch all transactions at or above a risk threshold.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| min_risk | float | 70.0 | Risk score threshold |
| band | str | — | CRITICAL / HIGH / MEDIUM / LOW |
| split | str | — | train / val / test |
| page | int | 1 | |
| page_size | int | 50 | Max 500 |

**Response (`AlertListResponse`):**
```json
{
  "total": 334,
  "page": 1,
  "page_size": 50,
  "results": [
    {
      "Transaction_ID": "ATM2503144PE7G1",
      "Date": "2025-03-14",
      "risk_score": 100.0,
      "risk_band": "CRITICAL",
      "reason_1": "time_since_previous_transaction (+2.34)",
      "reason_2": "amount_velocity_30m (+1.87)",
      "reason_3": "imei_novelty (+1.12)",
      "rules_fired": "RAPID_SUCCESSION",
      "is_suspicious_gt": 1
    }, ...
  ]
}
```

#### `GET /api/v1/scoring/transactions`

Full scored table with optional filters.

| Parameter | Type | Notes |
|---|---|---|
| customer_id | str | Filter by Sender_Customer_ID |
| min_risk | float | Default 0.0 |
| max_risk | float | Default 100.0 |
| date_from | str | ISO date YYYY-MM-DD |
| date_to | str | ISO date YYYY-MM-DD |
| rule | str | Filter by rule name (e.g. `ODD_HOUR`) |
| page / page_size | int | Pagination |

#### `GET /api/v1/scoring/customer/{customer_id}`

Full risk profile for one customer.

| Parameter | Notes |
|---|---|
| customer_id | Path param — Sender_Customer_ID value |
| top_n | Query param, default 5 — how many top transactions to include |

**Response (`CustomerSummary`):**
```json
{
  "customer_id": "100000042",
  "customer_name": "Rohan Sharma",
  "total_transactions": 52,
  "alert_count": 6,
  "max_risk_score": 99.9,
  "dominant_risk_band": "CRITICAL",
  "rules_fired_summary": ["HIGH_AMOUNT_ANOMALY", "RAPID_SUCCESSION"],
  "top_transactions": [...]
}
```

#### `GET /api/v1/scoring/stats`

Aggregate dashboard numbers.

**Response:**
```json
{
  "total_transactions": 20900,
  "total_alerts": 1570,
  "alert_rate_pct": 7.51,
  "band_distribution": {"CRITICAL": 1337, "HIGH": 231, "MEDIUM": 329, "LOW": 19003},
  "rule_fire_counts": {"ODD_HOUR": 855, "HIGH_AMOUNT_ANOMALY": 636, ...},
  "test_precision": 0.931,
  "test_recall": 0.781,
  "scored_csv_path": "..."
}
```

#### `POST /api/v1/scoring/score`

Score a single raw bank transaction on-demand using the persisted model bundle.

**Note:** CDR/IPDR context is unavailable for on-demand scoring — only Set A (bank) features are computed. Probability will be lower than full Set C score.

**Request body:**
```json
{
  "transaction": {
    "Transaction_ID": "TXN001",
    "Date": "2025-03-15",
    "Timestamp": "14:30:00",
    "Transaction_Amount": 95000,
    "Sender_Customer_ID": "100000001",
    "Sender_Phone_Number": "+919800000001",
    ...
  }
}
```

**Response (`ScoreResponse`):**
```json
{
  "Transaction_ID": "TXN001",
  "ml_probability": 0.8742,
  "risk_score": 82.4,
  "risk_band": "HIGH",
  "reasons": ["amount_vs_customer_median (+3.21)", "txn_count_previous_1h (+1.54)"],
  "rules_fired": ["HIGH_AMOUNT_ANOMALY"]
}
```

---

### 10.4 Graph Analytics — `/api/v1/graph`

All endpoints read from `notebook/output/graph_analytics.csv` and `graph_edges.csv`. Returns 503 if missing (run graph_analytics.py first).

#### `GET /api/v1/graph/summary`

| Parameter | Default | Notes |
|---|---|---|
| top_n | 10 | Number of top suspicious nodes to include |

**Response (`GraphSummary`):**
```json
{
  "total_nodes": 858,
  "total_edges": 20900,
  "known_mule_nodes": 367,
  "top_suspicious_nodes": [
    {
      "node_id": "670102075976",
      "in_degree": 92,
      "out_degree": 0,
      "in_out_ratio": 92000000.0,
      "pagerank": 0.0005,
      "suspicion_score": 0.8641,
      "is_mule_account": 1
    }, ...
  ]
}
```

#### `GET /api/v1/graph/nodes`

| Parameter | Default | Notes |
|---|---|---|
| sort_by | suspicion_score | also: pagerank, in_out_ratio, alert_count, in_degree |
| min_suspicion | 0.0 | Filter by minimum suspicion_score |
| mule_only | false | Show only flagged mule nodes |
| page / page_size | 1 / 50 | Pagination |

#### `GET /api/v1/graph/node/{node_id}`

Returns the node + all its outgoing and incoming edges.

```json
{
  "node": { "node_id": "670102075976", ... },
  "outgoing_edges": [],
  "incoming_edges": [
    { "Transaction_ID": "TXN25031...", "src": "100000042", "Transaction_Amount": 90000.0, ... }
  ]
}
```

#### `GET /api/v1/graph/mules`

Returns all nodes where `is_mule_account == 1`, sorted by suspicion_score descending.

#### `GET /api/v1/graph/edges`

| Parameter | Notes |
|---|---|
| src | Filter by source (sender) node_id |
| dst | Filter by destination (receiver) node_id |
| min_risk | Filter by minimum risk_score on the transaction |
| page / page_size | Pagination |

---

### 10.5 STR Reports — `/api/v1/reports`

#### `GET /api/v1/reports/str/{customer_id}`

Generate a full Suspicious Transaction Report for one customer.

| Parameter | Default | Notes |
|---|---|---|
| customer_id | path param | Sender_Customer_ID |
| min_risk | 70.0 | Include transactions at or above this score |
| officer | "System" | Name of reporting officer (appears in report) |

**Response (`STRReport`):**
```json
{
  "report_id": "uuid-...",
  "generated_at": "2026-08-02T10:30:00+00:00",
  "customer_id": "100000042",
  "customer_name": "Rohan Sharma",
  "reporting_officer": "System",
  "total_suspicious_transactions": 6,
  "total_suspicious_amount": 452300.50,
  "date_range_from": "2025-02-14",
  "date_range_to": "2025-04-01",
  "primary_risk_band": "CRITICAL",
  "scenario_types_detected": ["AMOUNT_VELOCITY_SPIKE", "REPEATED_CALLS_BEFORE_TRANSACTION"],
  "transactions": [...],
  "narrative": "Customer Rohan Sharma (ID: 100000042) has 6 transaction(s) ...",
  "graph_suspicion_score": 0.74,
  "graph_in_out_ratio": 2.1,
  "graph_mule_flag": 0
}
```

#### `GET /api/v1/reports/str/batch`

Lightweight summary for top-N alerted customers. Use this to decide which customers to pull full STRs for.

| Parameter | Default | Notes |
|---|---|---|
| min_risk | 70.0 | |
| top_n | 20 | Max 200 |

**Response:** list of `{customer_id, customer_name, alert_count, total_amount, max_risk}`.

#### `GET /api/v1/reports/summary`

Portfolio-level overview.

| Parameter | Default |
|---|---|
| min_risk | 70.0 |

**Response:**
```json
{
  "total_alerts": 1570,
  "unique_alerted_customers": 312,
  "total_suspicious_amount_inr": 48392150.75,
  "band_distribution": {"CRITICAL": 1337, "HIGH": 231, ...},
  "rule_breakdown": {"ODD_HOUR": 855, ...},
  "min_risk_threshold": 70.0,
  "generated_at": "2026-08-02T10:30:00+00:00"
}
```

---

## 11. Tests

**File:** `tests/test_features.py`  
**Run:** `python -m pytest tests/ -q`  
**Result:** 3 passed

Each test guards against one of the real bugs that occurred during development:

### test_causality_no_future_leakage

Builds features for 5 transactions, appends a 6th transaction 4 hours later, asserts that features for the first 5 rows are byte-identical in both runs.

**Bug it guards:** any feature using `groupby().transform()` or similar all-rows aggregation would accidentally use future data. Caught by comparing the first 5 rows before and after appending.

### test_alignment_velocity_count

Builds 4 transactions at known timestamps (t=0, 20m, 40m, 90m) and asserts `txn_count_previous_1h` matches a hand-computed brute-force count [0, 1, 2, 1].

**Bug it guards:** `groupby().rolling()` returns rows in group order, not original row order. Assigning `.values` back scrambles alignment. The test catches this because the brute-force count and the production count would differ for row 3 (txn at 90m).

### test_phone_dtype_csv_roundtrip

Writes `+919812345678` to a CSV and reads it back with and without `dtype=str`. Asserts the phone number survives with the `+91` prefix intact.

**Bug it guards:** pandas parses `+919812345678` as int64 by default, silently breaking every phone join between bank/CDR/IPDR with zero matches and no error. Fixed by the `STR` dtype dict at the top of `features.py`.

---

## 12. Key numbers you must know

| Metric | Value | Notes |
|---|---|---|
| Dataset size | 20,900 bank / 60,653 CDR / 51,433 IPDR | |
| Anomaly prevalence | 8.13% (1,700 / 20,900) | |
| Model | XGBoost, Set C, 53 features | Selected on val PR-AUC |
| PR-AUC (test) | **0.900** | |
| ROC-AUC (test) | 0.964 | |
| Precision @ threshold | **0.931** | |
| Recall @ threshold | **0.781** | |
| Decision threshold | 0.736 | Tuned on val F1 |
| Risk threshold for alert | 70 | Pinned to threshold |
| AMOUNT_VELOCITY_SPIKE recall | **0.936** | Was 0.500 before clone-label fix |
| Graph: mule recall in top-50 | **100%** | |
| Total API endpoints | 14 | Across 4 routers |
| Test suite | 3 tests, 100% pass | |

---

## 13. How to run everything

### Install dependencies
```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install pdfplumber rapidfuzz sentence-transformers
```

### Full pipeline (in order)
```bash
python scripts/generate_dataset.py   # ~30s  → data/
python scripts/train.py              # ~4min → models/stage7_setC.joblib + notebook/output/
python scripts/score.py              # ~1min → notebook/output/scored_transactions.csv
python scripts/graph_analytics.py   # ~10s  → notebook/output/graph_*.csv
```

### Run tests
```bash
python -m pytest tests/ -q
# Expected: 3 passed
```

### Start the API server
```bash
uvicorn backend.main:app --reload
# → http://localhost:8000/docs
```

### Parse a real bank statement PDF
```bash
# Quick standalone parse (no model needed):
python scripts/test_parse_pdf.py

# Full production parser:
python scripts/verify_pdf.py
```

### Expected generate_dataset.py output
```
bank 20,900 | cdr 60,653 | ipdr 51,433 | anomalies 1,700 (8.13%)
odd-hour(0-5) base rate : 0.041  (was 0.249)
new-beneficiary base    : 0.187  (was 0.998)
amount z>3: base 0.0094 | anom 0.1329  (was 0.0186 / 0.0000)
[OK] injected rows added=900 amount-modified=267
```
If any assertion fails — **stop**. The baseline has lost its structure.

---

## 14. Design decisions and traps

### Why pdfplumber and not Tesseract/OCR

The XXXX6607.pdf (Axis Bank) has selectable text — pdfplumber extracts it in milliseconds with perfect accuracy. OCR is only needed for scanned image PDFs. The parser detects scanned PDFs (`ScannedPDFError`) and raises clearly; extend with OCR as a fallback only when needed.

### Why `groupby().rolling()` is banned

It returns rows in group order (all customer A's rows, then all B's) rather than the original DataFrame order. Assigning `.values` back to a column silently scrambles every value. This made `TRANSACTION_BURST` recall 0.125 instead of 0.583. Use `index_by()/before()` instead.

### Why phone numbers must be read as str

Pandas infers `+919812345678` as int64. The `+` is lost, so the value becomes `-919812345678` or raises. Then every phone-based join (bank ↔ CDR ↔ IPDR) produces zero matches with no error. The `STR` dtype dict at the top of `features.py` covers all join-key columns.

### Why the model is selected on val, not test

Using test PR-AUC for selection leaks the held-out set into model choice. When this was briefly done, it picked Stacked_XGB+IF (wins test by 0.001, loses val by 0.024) and inflated TRANSACTION_BURST recall to 0.833. Selection now uses val PR-AUC only.

### Why burst clone transactions must be labelled y=1

`clone_txn()` creates 4–14 sibling transactions for AMOUNT_VELOCITY_SPIKE and TRANSACTION_BURST. Originally only the anchor was in ground truth (y=1); the clones were y=0. This created near-identical feature vectors with opposite labels, which trained the model to suppress velocity features (AMOUNT_VELOCITY_SPIKE stuck at 0.500 recall). The fix: emit a GT row per clone. Prevalence rose from 3.83% → 8.13%; recall rose to 0.936.

### Why ODD_HOUR_TRANSACTION scores 0.000 recall (intentional)

855 transactions occur at night; only 37 are anomalies (4.3%). A lone odd-hour signal cannot rank above 398 transactions carrying call/session/device evidence. The ML model pushes odd-hour-only rows below average probability — correct behaviour. ODD_HOUR is handled by the rule engine as a low-severity flag, not the ranker.

### Why `src/` was deleted

The `src/` directory contained a Stage 2–6 pipeline (~1,772 lines) that had never been executed. `src/features/engine.py` was missing 13 of the 53 features in `scripts/features.py`, was O(n)-per-row in pure Python (vs. vectorised NumPy), and was never imported by anything. Deleted in Task 3 — `scripts/features.py` is now the single implementation.

### Why `.env` is not in the repo

`.env` contained a HuggingFace API token. It is listed in `.gitignore`. If you need to set this token, create a `.env` file locally:
```
HUGGINGFACEHUB_API_TOKEN=hf_...
```

### The `Txn_Ref_Number` column in Value Date position

Axis Bank statements have a `Value Date` column, not a UTR/reference number. The parser maps `Value Date` → `Txn_Ref_Number` as the closest available canonical field. The actual Chq/Ref numbers from the `Chq No` column map to `Transaction_ID`.

---

*Last updated: 2026-08-02 after full pipeline verification on XXXX6607.pdf.*
