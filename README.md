# TRI-NETRA
**AI-Powered Financial & Telecom Dataset Analyzer (Bank, CDR, and IPDR Fusion)**

**Problem Statement ID:** ERH26_PS_03

## Project Vision
TRI-NETRA is an investigation-oriented financial and telecom data-fusion system designed to correlate:
- **Bank transactions**
- **Call Detail Records (CDR)**
- **Internet Protocol Detail Records (IPDR)**

The system helps investigators move from three large, heterogeneous datasets to a unified view of who transacted, who communicated, which device/subscriber identity was involved, what network activity occurred, and which events deserve investigation.

The complete system spans data ingestion, canonical normalization, entity resolution, cross-dataset correlation, unified timeline fusion, feature engineering, rules and machine learning, risk scoring, graph network analysis, and an investigation API for forensic STR reporting.

## Core Architectural Principles
- **Bank Transaction as the Risk Anchor:** One model observation equals one bank transaction. CDR and IPDR records provide contextual evidence around that transaction.
- **Fusion Is More Than an ML Join:** The fusion engine acts as a reusable system component supplying a unified timeline, ML features, and graph representations.
- **Rules and ML Work Together:** Known suspicious patterns are handled by a Rule Engine, while statistical and unusual behaviors are detected by a Machine Learning Engine.

## Repository Structure
```text
TRI-NETRA/
├── data/
│   ├── clean/             # Clean baseline datasets (bank, cdr, ipdr)
│   ├── anomalous/         # Datasets with controlled suspicious events injected
│   └── ground_truth/      # Ground truth labels for validation and evaluation
├── docs/                  # Detailed project documentation and specifications
├── notebook/              # Jupyter notebooks for exploration and analysis
│   └── output/            # Generated artefacts: scored_transactions.csv, graph_analytics.csv, etc.
├── scripts/               # Authoritative pipeline scripts
│   ├── generate_dataset.py   # Dataset builder (SEED=42, self-asserting)
│   ├── features.py           # THE feature builder — imported by train + score
│   ├── train.py              # Ablation + tuning; writes models/stage7_setC.joblib
│   ├── score.py              # Risk scores, TreeSHAP reasons, rule engine
│   └── graph_analytics.py    # Stage 9 graph analytics; writes graph_analytics.csv
├── models/
│   └── stage7_setC.joblib    # Persisted model bundle (XGBoost, 53 features)
├── src/                   # removed (Task 3 Option B — merged into scripts/features.py)
├── backend/               # FastAPI backend
│   ├── main.py               # App entry point — registers all routers
│   ├── pdf/                  # PDF parser endpoints
│   ├── scoring/              # Investigation search + on-demand scoring (Stage 10)
│   ├── graph/                # Graph analytics query endpoints (Stage 9)
│   └── reports/              # STR forensic report export (Stage 12)
├── tests/
│   └── test_features.py      # 3 passing tests (causality, alignment, dtype)
├── HANDOFF.md             # Authoritative state document — read before touching anything
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Stage-Wise Roadmap

| Stage | Component | Status |
|---|---|---|
| **1** | Dataset Preparation & Controlled Ground Truth | **DONE** |
| **2** | Canonical Internal Data Model | DONE (merged into `scripts/features.py`) |
| **3** | Entity Resolution | DONE (merged into `scripts/features.py`) |
| **4** | Cross-Dataset Correlation Engine | DONE (merged into `scripts/features.py`) |
| **5** | Unified Timeline & Fusion Layer | DONE (merged into `scripts/features.py`) |
| **6** | Feature Engineering | **DONE** (`scripts/features.py` — 53 causal features, Sets A/B/C) |
| **7** | Rules + ML Anomaly Detection | **DONE** (XGBoost Set C, PR-AUC 0.900, test precision 0.931) |
| **8** | Risk Scoring & Explainability | **DONE** (`scripts/score.py` — 0–100 score, TreeSHAP, 5-rule engine) |
| **9** | Graph / Network Analytics | **DONE** (`scripts/graph_analytics.py` — 100% mule recall in top-50) |
| **10** | Investigation Search / Backend API | **DONE** (`backend/scoring/router.py`) |
| **11** | Dashboard & Visualisation | Pending (wire frontend to existing API) |
| **12** | Forensic / STR Reporting | **DONE** (`backend/reports/router.py`) |
| **13** | Multi-Format & Provider-Specific Ingestion | Future |
| **14** | Scalability, Testing & Production Hardening | Future |

## Reproducing the pipeline

```bash
# from the repo root
pip install -r requirements.txt

python scripts/generate_dataset.py   # ~30 s — rewrites data/, self-asserts
python scripts/train.py              # ~4 min — writes models/ + notebook/output/
python scripts/score.py              # ~1 min — writes scored_transactions.csv
python scripts/graph_analytics.py    # ~10 s — writes graph_analytics.csv
python -m pytest tests/ -q           # 3 passed
```

Expected `generate_dataset.py` tail:
```
bank 20,900 | cdr 60,653 | ipdr 51,433 | anomalies 1,700 (8.13%)
[OK] injected rows added=900 amount-modified=267
```

## Running the backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
# → http://localhost:8000/docs   (Swagger UI)
```

## Key results (Set C XGBoost, test split)

| Metric | Value |
|---|---|
| PR-AUC | **0.900** |
| ROC-AUC | 0.964 |
| Precision @ threshold | 0.931 |
| Recall @ threshold | 0.781 |
| P@100 | 0.99 |
| AMOUNT_VELOCITY_SPIKE recall | **0.936** (was 0.500 before §3.4 fix) |
| Graph: mule recall in top-50 nodes | **100%** |

## Detailed Documentation
For detailed information about dataset structures, anomaly generation, and specific implementation requirements per stage, see [Stage-Wise Documentation](docs/TRI_NETRA_STAGE_WISE_DOCUMENTATION.md).

For the full engineering history, known issues, and task list, see [HANDOFF.md](HANDOFF.md).
