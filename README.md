# TRI-NETRA

**AI-powered financial fraud detection through Bank, CDR, and IPDR data fusion.**

Problem Statement ID: ERH26_PS_03

---

## What it does

TRI-NETRA takes three data sources that investigators already have — bank transaction records, call detail records (CDR), and internet session records (IPDR) — and fuses them to detect suspicious financial activity. Every alert comes with an explainability score, the exact features that drove the prediction, and a ready-to-file Suspicious Transaction Report (STR).

The system detects 15 fraud scenarios including amount velocity spikes, unusual call patterns before high-value transfers, new device appearances, location anomalies, and structured layering through mule accounts.

---

## Key results

| Metric | Value |
|---|---|
| PR-AUC (test split) | **0.900** |
| Precision at threshold | **0.931** |
| Recall at threshold | **0.781** |
| AMOUNT_VELOCITY_SPIKE recall | **0.936** |
| Mule account detection (top-50 nodes) | **100%** |
| Fraud scenarios detected | **14 of 15** |

---

## Project structure

```
TRI-NETRA/
├── backend/                  Python pipeline + FastAPI
│   ├── main.py               API entry point
│   ├── scripts/              ML pipeline scripts
│   │   ├── generate_dataset.py
│   │   ├── features.py       Single feature implementation (53 features)
│   │   ├── train.py          Model training and ablation
│   │   ├── score.py          Risk scoring + rule engine
│   │   └── graph_analytics.py
│   ├── models/               Trained model bundle (.joblib)
│   ├── notebook/output/      Generated CSVs (scores, graph, STR)
│   ├── pdf-parser/           Bank statement / CDR / IPDR PDF parser
│   ├── pdf/                  PDF upload API module
│   ├── scoring/              Investigation search API
│   ├── graph/                Graph analytics API
│   ├── reports/              STR report export API
│   └── tests/                3 regression tests
├── frontend/                 React + Vite investigation dashboard
│   └── src/
│       ├── pages/            Dashboard, Alerts, Graph, Reports, PDF Parser
│       └── components/       Shared UI components
├── data/                     Datasets (git-tracked)
│   ├── clean/                Baseline (20k bank, 60k CDR, 50k IPDR)
│   ├── anomalous/            Baseline + 1,700 injected anomalies
│   └── ground_truth/         Labels + correlation ground truth
└── docs/                     Stage-wise technical documentation
```

---

## Quick start

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Install dependencies

```bash
# Backend
cd backend
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib
pip install pdfplumber rapidfuzz sentence-transformers python-multipart

# Frontend
cd ../frontend
npm install
```

### 2. Run the ML pipeline (one time)

```bash
cd backend

# Build the dataset
python scripts/generate_dataset.py

# Train the model (~4 min)
python scripts/train.py

# Score all transactions
python scripts/score.py

# Build the graph
python scripts/graph_analytics.py
```

### 3. Start the application

Open two terminals:

**Terminal 1 — Backend**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm run dev
```

| Service | URL |
|---|---|
| Investigation Dashboard | http://localhost:3000 |
| API + Swagger UI | http://localhost:8000/docs |

---

## API overview

The backend exposes 14 endpoints across 4 modules.

**Scoring** — search and filter all 20,900 scored transactions, drill into individual customers, score a new transaction on demand.

**Graph** — query the financial transfer graph (858 nodes, 20,900 edges), list mule accounts by suspicion score, inspect node-level evidence.

**Reports** — generate structured STR reports per customer with auto-written narrative, detected scenario types, and graph context. Export as JSON.

**PDF Parser** — upload any bank statement, CDR, or IPDR PDF. The parser auto-detects the document type, extracts all records into the canonical schema, and returns structured JSON. Supports Axis Bank, HDFC, SBI, ICICI, and others.

Full route documentation: **http://localhost:8000/docs**

---

## How the model works

Each bank transaction is one observation. CDR and IPDR records provide context around it — they are never observations themselves.

The pipeline builds 53 causal features across three sets:

- **Set A (25 features)** — bank only: amount velocity, customer-relative spike, new beneficiary, hour rarity
- **Set B (+17 features)** — adds CDR: call patterns before transaction, device novelty, location mismatch
- **Set C (+11 features)** — adds IPDR: session bursts, IP novelty, IMSI/IMEI pair changes

XGBoost is trained on Set C with a temporal split (60% train / 15% val / 25% test). The threshold is tuned on the validation set. Every flagged transaction carries exact TreeSHAP reason codes.

A rule engine runs independently of the ML model and flags signals the ranker cannot rank reliably on its own (odd-hour transactions, telecom bursts, rapid succession, etc.).

---

## Run the tests

```bash
cd backend
python -m pytest tests/ -q
```

3 tests covering causality (no future data leakage), velocity count alignment, and phone number dtype preservation.

---

## Tech stack

| Layer | Technology |
|---|---|
| ML | XGBoost, scikit-learn, pandas, NumPy |
| Explainability | TreeSHAP (via XGBoost's `pred_contribs`) |
| Graph analytics | Custom power-iteration PageRank, no external graph library |
| Backend | FastAPI, Uvicorn |
| PDF parsing | pdfplumber, RapidFuzz, SentenceTransformers |
| Frontend | React 18, Vite 5, Recharts, React Router v6 |
| Model persistence | joblib |

---

## Detailed documentation

- [`HANDOFF.md`](HANDOFF.md) — full engineering history, verified numbers, known issues, design decisions
- [`CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md) — complete reference for every file, feature, API route, and design decision
- [`docs/TRI_NETRA_STAGE_WISE_DOCUMENTATION.md`](docs/TRI_NETRA_STAGE_WISE_DOCUMENTATION.md) — stage specifications

---

## License

[MIT](LICENSE)
