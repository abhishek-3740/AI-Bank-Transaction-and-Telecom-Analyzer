# TRI-NETRA
**AI-Powered Financial & Telecom Dataset Analyzer (Bank, CDR, and IPDR Fusion)**

**Problem Statement ID:** ERH26_PS_03

## 📖 Project Vision
TRI-NETRA is an investigation-oriented financial and telecom data-fusion system designed to correlate:
- **Bank transactions**
- **Call Detail Records (CDR)**
- **Internet Protocol Detail Records (IPDR)**

The system helps investigators move from three large, heterogeneous datasets to a unified view of who transacted, who communicated, which device/subscriber identity was involved, what network activity occurred, and which events deserve investigation. 

The complete system spans data ingestion, canonical normalization, entity resolution, cross-dataset correlation, unified timeline fusion, feature engineering, rules and machine learning, risk scoring, graph network analysis, and an investigation dashboard for forensic STR reporting.

## 🏗️ Core Architectural Principles
- **Bank Transaction as the Risk Anchor:** One model observation equals one bank transaction. CDR and IPDR records provide contextual evidence around that transaction.
- **Fusion Is More Than an ML Join:** The fusion engine acts as a reusable system component supplying a unified timeline, ML features, and graph representations.
- **Rules and ML Work Together:** Known suspicious patterns are handled by a Rule Engine, while statistical and unusual behaviors are detected by a Machine Learning Engine.

## 📂 Repository Structure
```text
TRI-NETRA/
├── data/
│   ├── clean/            # Clean baseline datasets (bank, cdr, ipdr)
│   ├── anomalous/        # Datasets with controlled suspicious events injected
│   └── ground_truth/     # Ground truth labels for validation and evaluation
├── docs/                 # Detailed project documentation and specifications
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── src/                  # Source code for the application and models
├── tests/                # Unit and integration tests
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## 🚀 Stage-Wise Roadmap

| Stage | Component | Status |
|---|---|---|
| **1** | Dataset Preparation & Controlled Ground Truth | **COMPLETED** |
| **2** | Canonical Internal Data Model | **NEXT** |
| **3** | Entity Resolution | Pending |
| **4** | Cross-Dataset Correlation Engine | Pending |
| **5** | Unified Timeline & Fusion Layer | Pending |
| **6** | Feature Engineering | Pending |
| **7** | Rules + ML Anomaly Detection | Pending |
| **8** | Risk Scoring & Explainability | Pending |
| **9** | Graph / Network Analytics | Pending |
| **10** | Investigation Search / Backend API | Pending |
| **11** | Dashboard & Visualisation | Pending |
| **12** | Forensic / STR Reporting | Pending |
| **13** | Multi-Format & Provider-Specific Ingestion | Future |
| **14** | Scalability, Testing & Production Hardening | Future |

## 📚 Detailed Documentation
For detailed information about dataset structures, anomaly generation, and specific implementation requirements per stage, please refer to the comprehensive [Stage-Wise Documentation](docs/TRI_NETRA_STAGE_WISE_DOCUMENTATION.md) located in the `docs/` folder.
