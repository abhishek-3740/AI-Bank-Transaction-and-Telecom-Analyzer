# TRI-NETRA

## Stage-Wise Development Roadmap and Technical Documentation

**Problem Statement ID:** ERH26_PS_03\
**Problem:** AI-Powered Financial & Telecom Dataset Analyzer (Bank, CDR,
and IPDR Fusion)\
**Domain:** Big Data and Analytics\
**Current Status:** **Stage 1 Completed**\
**Next Stage:** **Stage 2 --- Canonical Internal Data Model**

------------------------------------------------------------------------

# 1. Project Vision

TRI-NETRA is an investigation-oriented financial and telecom data-fusion
system designed to correlate:

-   Bank transactions
-   Call Detail Records (CDR)
-   Internet Protocol Detail Records (IPDR)

The system should help investigators move from three large,
heterogeneous datasets to a unified view of:

-   who transacted;
-   who communicated;
-   which device/subscriber identity was involved;
-   what network activity occurred;
-   when these events happened;
-   how entities are connected;
-   which transactions or entities deserve investigation.

The project is therefore **not only an anomaly-detection model**.

The complete system is:

``` text
Data Ingestion
      ↓
Canonical Normalisation
      ↓
Entity Resolution
      ↓
Cross-Dataset Correlation
      ↓
Unified Timeline / Fusion
      ↓
Feature Engineering
      ↓
Rules + Machine Learning
      ↓
Risk Scoring
      ↓
Graph / Network Analysis
      ↓
Investigation Dashboard
      ↓
Forensic / STR Reporting
```

------------------------------------------------------------------------

# 2. Current Development Strategy

The official problem statement requires heterogeneous ingestion from
CSV, Excel, PDF, and telecom-provider exports.

For the current development phase, this ingestion problem is
intentionally postponed.

We have already created controlled Bank, CDR, and IPDR CSV datasets with
known relationships and known suspicious events.

Therefore the current development boundary is:

``` text
PDF / Excel / Provider-specific exports
                │
                │  FUTURE
                ▼
        Parsing / Schema Mapping
                │
                ▼
        Canonical Data Model
                │
                │
         CURRENT DEVELOPMENT
                ▼
       Entity Resolution
                ↓
       Correlation / Fusion
                ↓
       Detection / Analytics
```

This lets us build and validate the core intelligence layer first.

Later, real parsers will simply convert external files into the same
canonical structures.

------------------------------------------------------------------------

# 3. Core Architectural Principles

## 3.1 Bank Transaction as the Risk Anchor

For suspicious-transaction modelling:

``` text
ONE MODEL OBSERVATION = ONE BANK TRANSACTION
```

CDR and IPDR records provide contextual evidence around that
transaction.

This does **not** mean CDR/IPDR are discarded. Event-level records
remain available for timelines, graphs, searches, and forensic evidence.

------------------------------------------------------------------------

## 3.2 Fusion Is More Than an ML Join

The fusion engine must support:

``` text
                     FUSION
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     Timeline      ML Features      Graph
        │              │              │
        ▼              ▼              ▼
 Investigation    Risk Detection   Networks
```

Fusion must therefore be implemented as a reusable system component
rather than a temporary CSV merge.

------------------------------------------------------------------------

## 3.3 Ground Truth Is Never a Predictive Feature

The following files are answer keys:

``` text
anomaly_ground_truth.csv
bank_cdr_ground_truth.csv
cdr_ipdr_ground_truth.csv
```

They are used for:

-   validation;
-   supervised labels;
-   benchmarking;
-   final evaluation.

They must **not** be used as shortcuts by the production correlation or
feature-generation pipeline.

------------------------------------------------------------------------

## 3.4 Temporal Leakage Must Be Prevented

For a transaction occurring at time `T`, historical behavioural features
should normally use information available before `T`.

Example:

``` text
Past behaviour ─────────────► T
                              │
                              ▼
                         Transaction
```

Future transactions must not be used to calculate what was supposedly
known about the customer at `T`.

------------------------------------------------------------------------

## 3.5 Rules and ML Work Together

The problem statement explicitly requires **Rules + ML**.

The final architecture should therefore support:

``` text
Known suspicious patterns
          ↓
      Rule Engine
          │
          ├─────────┐
          │         │
          │         ▼
          │      Risk Engine
          │         ▲
          │         │
          └─────────┤
                    │
Statistical / unusual behaviour
                    ↓
                 ML Engine
```

LLMs are not required for the core anomaly-detection path.

------------------------------------------------------------------------

# 4. Clean Repository Starting Point

``` text
TRI-NETRA/
│
├── data/
│   ├── clean/
│   │   ├── bank_final.csv
│   │   ├── cdr_final.csv
│   │   └── ipdr_final.csv
│   │
│   ├── anomalous/
│   │   ├── bank_anomaly.csv
│   │   ├── cdr_anomaly.csv
│   │   └── ipdr_anomaly.csv
│   │
│   └── ground_truth/
│       ├── anomaly_ground_truth.csv
│       ├── bank_cdr_ground_truth.csv
│       └── cdr_ipdr_ground_truth.csv
│
├── notebooks/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

Dataset-generation scripts and historical experiments remain in the
old/archive repository.

------------------------------------------------------------------------

# 5. Stage-Wise Roadmap

  -----------------------------------------------------------------------
  Stage                   Component               Status
  ----------------------- ----------------------- -----------------------
  **1**                   Dataset Preparation &   **COMPLETED**
                          Controlled Ground Truth 

  **2**                   Canonical Internal Data **NEXT**
                          Model                   

  **3**                   Entity Resolution       Pending

  **4**                   Cross-Dataset           Pending
                          Correlation Engine      

  **5**                   Unified Timeline &      Pending
                          Fusion Layer            

  **6**                   Feature Engineering     Pending

  **7**                   Rules + ML Anomaly      Pending
                          Detection               

  **8**                   Risk Scoring &          Pending
                          Explainability          

  **9**                   Graph / Network         Pending
                          Analytics               

  **10**                  Investigation Search /  Pending
                          Backend API             

  **11**                  Dashboard &             Pending
                          Visualisation           

  **12**                  Forensic / STR          Pending
                          Reporting               

  **13**                  Multi-Format &          Future
                          Provider-Specific       
                          Ingestion               

  **14**                  Scalability, Testing &  Future
                          Production Hardening    
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 6. STAGE 1 --- Dataset Preparation & Controlled Ground Truth

## Status: COMPLETED

Stage 1 establishes the controlled experimental environment required for
all later development.

------------------------------------------------------------------------

## 6.1 Bank Dataset

The Bank dataset contains transaction-centric financial information.

Important fields include:

``` text
Transaction_ID
Date
Timestamp
Txn_Ref_Number
Transaction_Mode
Currency
Transaction_Amount

Sender_Customer_ID
Sender_Customer_Name
Sender_Bank_Name
Sender_Account_Number
Sender_Account_Type
Sender_IFSC
Sender_Phone_Number

Receiver_Customer_ID
Receiver_Customer_Name
Receiver_Bank_Name
Receiver_Account_Number
Receiver_Account_Type
Receiver_IFSC
Receiver_Phone_Number
```

------------------------------------------------------------------------

## 6.2 CDR Dataset

The CDR dataset contains telecom communication activity.

Core schema:

``` text
CDR_ID
Call_Date
Call_Start_Time
A_Party_Number
B_Party_Number
Call_Type
Call_Duration_Seconds
IMSI
IMEI
First_BTS_Location
First_Cell_Global_ID
Roaming_Network_Circle
```

------------------------------------------------------------------------

## 6.3 IPDR Dataset

The IPDR dataset contains internet/network-session activity.

Core schema:

``` text
IPDR_ID
Session_Date
Session_Start_Time
Subscriber_IMSI
Subscriber_MSISDN
Device_IMEI
Source_IP_Address
Destination_IP_Address
Destination_Port
Cell_Global_ID
Session_Duration_Seconds
```

------------------------------------------------------------------------

## 6.4 Clean Baselines

The immutable baseline datasets are:

``` text
data/clean/bank_final.csv
data/clean/cdr_final.csv
data/clean/ipdr_final.csv
```

These represent the clean synthetic environment before controlled
anomaly injection.

------------------------------------------------------------------------

## 6.5 Anomaly-Injected Datasets

The modelling/investigation datasets are:

``` text
data/anomalous/bank_anomaly.csv
data/anomalous/cdr_anomaly.csv
data/anomalous/ipdr_anomaly.csv
```

These contain:

``` text
NORMAL RECORDS
      +
CONTROLLED SUSPICIOUS EVENTS
```

They are not anomaly-only datasets.

------------------------------------------------------------------------

## 6.6 Final Anomaly Population

The final controlled experiment contains:

``` text
Suspicious Bank transactions: 100
Approximate anomaly rate:      0.10%
Unique suspicious customers:   99
```

Difficulty distribution:

``` text
EASY      25
MEDIUM    35
HARD      40
```

Evidence-source distribution:

``` text
BANK_ONLY          25
BANK_CDR           30
BANK_CDR_IPDR      45
```

------------------------------------------------------------------------

## 6.7 Fifteen Suspicious Scenario Families

The dataset contains the following controlled scenario families:

1.  Customer-relative amount spike
2.  Odd-hour transaction
3.  New beneficiary
4.  Transaction burst
5.  Amount velocity spike
6.  Amount plus new beneficiary
7.  Unusual call before transaction
8.  Repeated calls before transaction
9.  Call followed by high-value transfer
10. Call followed by new-beneficiary transfer
11. New device around transaction
12. Unusual location context
13. IMSI-IMEI pair novelty
14. Network-session burst around transaction
15. Subtle multi-source suspicious pattern

The scenarios were designed around structured signals that conventional
algorithms can detect.

------------------------------------------------------------------------

## 6.8 Ground Truth Architecture

### Anomaly Ground Truth

``` text
data/ground_truth/anomaly_ground_truth.csv
```

Contains 100 primary suspicious Bank transactions.

It records information such as:

``` text
Anomaly_ID
Customer_ID
Transaction_ID
CDR_IDs
IPDR_IDs
Scenario_Type
Difficulty
Source_Scope
Injected_Signals
Is_Suspicious
```

It is the primary truth source for anomaly evaluation.

------------------------------------------------------------------------

### Bank ↔ CDR Ground Truth

``` text
data/ground_truth/bank_cdr_ground_truth.csv
```

Defines known relationships between Bank transactions and CDR activity.

Relationship types include concepts such as:

``` text
DIRECT
REVERSE
SENDER_INCOMING_ACTIVITY
SENDER_OUTGOING_ACTIVITY
RECEIVER_INCOMING_ACTIVITY
RECEIVER_OUTGOING_ACTIVITY
NO_MATCH
```

------------------------------------------------------------------------

### CDR ↔ IPDR Ground Truth

``` text
data/ground_truth/cdr_ipdr_ground_truth.csv
```

Defines known relationships between CDR records and IPDR sessions.

It supports one-to-many relationships such as:

``` text
CDR
 │
 ├── PRIMARY_SESSION
 ├── ADDITIONAL_SESSION
 └── ADDITIONAL_SESSION
```

------------------------------------------------------------------------

## 6.9 Stage 1 Validation

The prepared datasets have already passed the important sanity checks:

``` text
Bank → CDR semantic linkage          PASS
CDR → IPDR semantic linkage          PASS
Complete chain recoverability        PASS
Referential integrity                PASS
Anomaly detectability                PASS
Single-feature leakage audit         PASS
Ground-truth integrity               PASS
```

The datasets also use consistent:

-   identifiers;
-   timestamps;
-   phone/MSISDN formats;
-   IMSI values;
-   IMEI values;
-   Cell Global IDs.

------------------------------------------------------------------------

## 6.10 Stage 1 Freeze Rule

**Stage 1 is now frozen.**

Do not casually modify:

``` text
bank_final.csv
cdr_final.csv
ipdr_final.csv

bank_anomaly.csv
cdr_anomaly.csv
ipdr_anomaly.csv

anomaly_ground_truth.csv
bank_cdr_ground_truth.csv
cdr_ipdr_ground_truth.csv
```

If the synthetic environment must later change, create a new dataset
version.

Do not silently overwrite the current benchmark.

------------------------------------------------------------------------

# 7. STAGE 2 --- Canonical Internal Data Model

## Status: NEXT

## Goal

Define the internal representation TRI-NETRA uses regardless of the
original provider/file schema.

For example:

``` text
Sender_Phone_Number
Customer_Mobile
Mobile_No
```

may all eventually mean:

``` text
sender_phone
```

The correlation engine should operate on the canonical meaning, not
provider-specific labels.

------------------------------------------------------------------------

## 7.1 Required Internal Models

At minimum:

``` text
BankTransaction
CDREvent
IPDRSession
Entity
```

Potential entity identities:

``` text
Customer
Bank Account
Phone / MSISDN
IMSI
IMEI / Device
IP Address
Cell Global ID
Beneficiary
```

------------------------------------------------------------------------

## 7.2 Provenance

Canonicalisation should not destroy source information.

A record should eventually retain concepts such as:

``` text
source_type
source_file
source_record_id
canonical_timestamp
```

This becomes important for forensic traceability.

------------------------------------------------------------------------

## 7.3 Stage 2 Deliverable

A stable schema/model module inside `src/`.

Stage 2 does **not** require:

-   ML;
-   graph databases;
-   dashboard code;
-   PDF parsing.

------------------------------------------------------------------------

## 7.4 Stage 2 Exit Criteria

Stage 2 is complete when all three current datasets can be mapped
deterministically into canonical records while preserving everything
needed by downstream correlation and investigation.

------------------------------------------------------------------------

# 8. STAGE 3 --- Entity Resolution

## Status: Pending

## Goal

Resolve equivalent identities across datasets.

### Bank ↔ CDR

``` text
Bank.Sender_Phone_Number
Bank.Receiver_Phone_Number

            ↕

CDR.A_Party_Number
CDR.B_Party_Number
```

### CDR ↔ IPDR

``` text
CDR.IMSI
        ↕
IPDR.Subscriber_IMSI

CDR.IMEI
        ↕
IPDR.Device_IMEI

CDR Phone
        ↕
IPDR.Subscriber_MSISDN

CDR.First_Cell_Global_ID
        ↕
IPDR.Cell_Global_ID
```

------------------------------------------------------------------------

## 8.1 Initial Strategy

Begin with deterministic identity normalisation and matching.

Do not introduce fuzzy ML entity resolution unless the data actually
requires it.

Possible normalisation:

``` text
phone number formatting
country-code handling
string trimming
identifier type validation
timestamp normalisation
```

------------------------------------------------------------------------

## 8.2 Stage 3 Exit Criteria

Identity relationships must be recoverable without reading relationship
ground truth as input.

------------------------------------------------------------------------

# 9. STAGE 4 --- Cross-Dataset Correlation Engine

## Status: Pending

This is a core project stage.

Entity resolution answers:

> Are these identifiers equivalent?

Correlation answers:

> Are these events meaningfully related?

------------------------------------------------------------------------

## 9.1 Bank → CDR Correlation

For every Bank transaction:

``` text
BANK TRANSACTION
       │
       ├── sender phone
       ├── receiver phone
       └── transaction time
       │
       ▼
Search relevant CDR events
       │
       ├── A-party
       ├── B-party
       └── temporal proximity
```

The current synthetic environment supports approximately a ±30-minute
relationship window.

The implementation should make temporal windows configurable rather than
hard-coded throughout the codebase.

------------------------------------------------------------------------

## 9.2 Multiple CDR Matches

A Bank transaction can have:

``` text
0 CDR matches
1 CDR match
many CDR matches
```

All cases must be supported.

Do not discard additional calls merely to force a one-to-one
relationship.

------------------------------------------------------------------------

## 9.3 CDR → IPDR Correlation

Relevant CDRs are then correlated with IPDR sessions using:

``` text
IMSI
IMEI
MSISDN
Cell Global ID
Time
```

Multiple IPDR sessions may belong to one CDR context.

------------------------------------------------------------------------

## 9.4 Correlation Output

A useful correlation record should eventually contain:

``` text
source_event_id
target_event_id
relationship_type
time_difference_seconds
identity_evidence
match_confidence
```

------------------------------------------------------------------------

## 9.5 Validation

The algorithm must generate links independently.

Then compare them with:

``` text
bank_cdr_ground_truth.csv
cdr_ipdr_ground_truth.csv
```

Evaluate:

``` text
Precision
Recall
F1
False matches
Missed matches
```

------------------------------------------------------------------------

## 9.6 Stage 4 Exit Criteria

The correlation engine must accurately recover the intended
relationships while correctly supporting unmatched and one-to-many
cases.

------------------------------------------------------------------------

# 10. STAGE 5 --- Unified Timeline & Fusion Layer

## Status: Pending

## Goal

Create a unified chronological view across financial, telecom, and
network evidence.

Example:

``` text
CUSTOMER X

13:42:10   IPDR   Internet session
13:48:31   CDR    Incoming call
13:54:02   CDR    Call ended
13:57:14   IPDR   Network session
14:01:43   BANK   Transfer
14:03:12   IPDR   Network session
14:07:52   CDR    Outgoing call
```

This timeline itself is an important investigation feature even before
ML.

------------------------------------------------------------------------

## 10.1 Event-Level Fusion

Retain raw correlated events for:

-   timelines;
-   drill-down;
-   forensic evidence;
-   graph construction.

------------------------------------------------------------------------

## 10.2 Transaction-Centric Fusion

For modelling:

``` text
ONE BANK TRANSACTION
        │
        ├── related CDR context
        │
        └── related IPDR context
        │
        ▼
ONE TRANSACTION-CENTRIC CASE
```

Do not create an uncontrolled Cartesian join.

------------------------------------------------------------------------

## 10.3 Missing Context

Transactions with no telecom/network context must remain present.

Example:

``` text
has_cdr_context  = 0
has_ipdr_context = 0
```

Absence of context is itself information.

------------------------------------------------------------------------

# 11. STAGE 6 --- Feature Engineering

## Status: Pending

Raw IDs and raw records are generally not enough for anomaly detection.

The model needs behavioural signals.

------------------------------------------------------------------------

## 11.1 Bank Features

Examples:

``` text
amount_vs_customer_median
amount_robust_zscore
amount_percentile

receiver_seen_before
receiver_historical_count
receiver_frequency

transaction_hour
hour_rarity

txn_count_previous_10m
txn_count_previous_30m
txn_count_previous_1h

amount_velocity_30m
amount_velocity_1h

time_since_previous_transaction
```

------------------------------------------------------------------------

## 11.2 CDR Features

Examples:

``` text
has_cdr_context

calls_previous_10m
calls_previous_30m
calls_previous_1h

nearest_call_before_seconds

total_call_duration_30m
max_call_duration_30m

caller_novelty
caller_historical_frequency

imei_novelty
cell_novelty
roaming_change
```

------------------------------------------------------------------------

## 11.3 IPDR Features

Examples:

``` text
has_ipdr_context

sessions_previous_10m
sessions_previous_30m

nearest_session_seconds

source_ip_novelty
destination_ip_novelty
destination_port_novelty

imsi_imei_pair_novelty
device_consistency
cell_consistency

session_duration_deviation
```

------------------------------------------------------------------------

## 11.4 Feature Sets for Controlled Comparison

Prepare three matrices:

``` text
FEATURE SET A
Bank only

FEATURE SET B
Bank + CDR

FEATURE SET C
Bank + CDR + IPDR
```

This comparison is central to demonstrating the value of fusion.

------------------------------------------------------------------------

## 11.5 Leakage Testing

For each transaction at time `T`, behavioural baselines must not use
inappropriate future records.

Feature-generation code must receive dedicated leakage tests.

------------------------------------------------------------------------

# 12. STAGE 7 --- Rules + ML Anomaly Detection

## Status: Pending

## Goal

Assign suspiciousness to transactions based on structured evidence.

------------------------------------------------------------------------

## 12.1 Rule Engine

Rules handle known, interpretable patterns.

Examples:

``` text
rapid transaction activity
rapid money movement
large customer-relative amount deviation
new beneficiary + unusual activity
call immediately preceding transfer
device/location inconsistency
```

Later graph-aware rules can detect:

``` text
layering
structuring
circular flows
mule-account signatures
```

------------------------------------------------------------------------

## 12.2 ML Models

Do not begin with unnecessary deep-learning complexity.

Recommended initial models:

### Isolation Forest

Primary unsupervised anomaly-detection baseline.

Suitable because the suspicious population is extremely rare.

### Random Forest

Simple supervised benchmark.

### XGBoost / LightGBM-style Gradient Boosting

Strong supervised benchmark for structured/tabular features.

------------------------------------------------------------------------

## 12.3 Supervised Labels

For controlled experiments:

``` text
Transaction_ID in anomaly_ground_truth
        ↓
y = 1

otherwise
        ↓
y = 0
```

The model must never receive ground-truth metadata such as:

``` text
Scenario_Type
Difficulty
Anomaly_ID
Injected_Signals
Is_Suspicious
```

as predictive features.

------------------------------------------------------------------------

## 12.4 Evaluation Metrics

Accuracy is not an appropriate headline metric for approximately 0.10%
anomalies.

Use:

``` text
Precision
Recall
F1
PR-AUC
ROC-AUC
False Positive Rate
Precision@K
Recall@K
```

------------------------------------------------------------------------

## 12.5 Scenario Coverage

The final detector should be evaluated across all 15 scenario families.

The model is **not** a 15-class scenario classifier.

It should learn underlying suspicious behavioural signals and
potentially detect unseen combinations of them.

------------------------------------------------------------------------

# 13. STAGE 8 --- Risk Scoring & Explainability

## Status: Pending

The final system should not merely output:

``` text
1
```

or:

``` text
FRAUD
```

Instead it should produce an investigator-oriented risk assessment.

Example:

``` text
Transaction Risk: HIGH

Contributing Evidence:
- amount significantly exceeds customer baseline;
- beneficiary is new;
- unusual call occurred shortly before transfer;
- device identity is unusual;
- network location differs from historical behaviour.
```

The system should describe activity as **suspicious** rather than
automatically declaring criminality.

------------------------------------------------------------------------

# 14. STAGE 9 --- Graph / Network Analytics

## Status: Pending

Transaction-level ML cannot solve all requirements in the problem
statement.

Money-flow and communication networks require graph analysis.

------------------------------------------------------------------------

## 14.1 Possible Nodes

``` text
Customer
Account
Transaction
Phone
CDR Event
IMSI
IMEI / Device
IP Address
Cell
IPDR Session
Beneficiary
```

------------------------------------------------------------------------

## 14.2 Possible Edges

``` text
SENT_TO
RECEIVED_FROM
CALLED
CALLED_BY
USES_PHONE
USES_DEVICE
USES_IMSI
CONNECTED_FROM_IP
OBSERVED_AT_CELL
CORRELATED_WITH
```

------------------------------------------------------------------------

## 14.3 Graph Use Cases

Graph analysis can support:

-   mule-account detection;
-   circular transfers;
-   layering;
-   shared-device networks;
-   shared-phone networks;
-   communication clusters;
-   cross-bank networks;
-   cross-operator networks;
-   high-risk hubs.

Start with NetworkX.

Use Neo4j later only if persistent graph querying and scale justify it.

------------------------------------------------------------------------

# 15. STAGE 10 --- Investigation Search / Backend API

## Status: Pending

Provide backend operations for:

``` text
search by customer
search by account
search by phone
search by transaction
search by IMSI
search by IMEI
search by IP
filter by amount
filter by date/time
filter by location
filter by risk
retrieve unified timeline
retrieve correlated evidence
retrieve network neighbourhood
```

The frontend should consume stable APIs rather than directly
manipulating CSV files.

------------------------------------------------------------------------

# 16. STAGE 11 --- Dashboard & Visualisation

## Status: Pending

The dashboard should focus on investigator workflow.

Core views:

### Search

Find entities and transactions.

### Unified Timeline

Show Bank, CDR, and IPDR events together.

### Suspicious Transaction Queue

Rank events by risk.

### Evidence Drill-Down

Display why a transaction is suspicious.

### Network Graph

Visualise money and communication connections.

### Filters

Filter by:

``` text
entity
amount
time
location
risk
event type
```

------------------------------------------------------------------------

# 17. STAGE 12 --- Forensic / STR Reporting

## Status: Pending

Generate investigation-ready reports containing:

-   case metadata;
-   suspicious transactions;
-   relevant entities;
-   evidentiary timeline;
-   matched CDR evidence;
-   matched IPDR evidence;
-   risk indicators;
-   money-flow visualisation;
-   communication/network visualisation;
-   methodology/provenance.

Automated STR generation is a bonus capability and should be implemented
only after upstream evidence is reliable.

------------------------------------------------------------------------

# 18. STAGE 13 --- Multi-Format and Provider-Specific Ingestion

## Status: FUTURE

This stage intentionally comes later in the current development plan.

Eventually:

``` text
PDF ────┐
Excel ──┼──► Parser / Schema Mapper ──► Canonical Model
CSV ────┘
```

For telecom:

``` text
Operator A ──┐
Operator B ──┼──► Provider Adapter ──► Canonical CDR/IPDR
Operator C ──┘
```

Potential technologies:

``` text
Pandas
OpenPyXL
pdfplumber / equivalent PDF extraction
schema detection
data-quality validation
```

The downstream fusion engine must remain unchanged regardless of input
provider.

------------------------------------------------------------------------

# 19. STAGE 14 --- Scalability, Testing & Production Hardening

## Status: FUTURE

The final system must be tested beyond the current synthetic scale.

Potential targets:

``` text
100K rows
1M rows
multi-million-row workloads
```

Potential optimisation techniques:

-   vectorised operations;
-   temporal indexes;
-   entity/date partitioning;
-   Polars or DuckDB where justified;
-   PostgreSQL indexes;
-   caching;
-   batch processing;
-   graph optimisation.

Testing should include:

``` text
unit tests
schema tests
entity-resolution tests
correlation tests
temporal-boundary tests
feature-leakage tests
model tests
graph tests
integration tests
end-to-end tests
```

------------------------------------------------------------------------

# 20. Final Target Architecture

``` text
                   EXTERNAL DATA
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        BANK            CDR            IPDR
          │              │              │
          └──────────────┼──────────────┘
                         ▼
              INGESTION / PARSING
                    [FUTURE]
                         │
                         ▼
                CANONICAL MODEL
                         │
                         ▼
                ENTITY RESOLUTION
                         │
                         ▼
               CORRELATION ENGINE
                         │
                         ▼
                  FUSION LAYER
                         │
                         ▼
                 UNIFIED TIMELINE
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       FEATURES         GRAPH        SEARCH/API
          │
     ┌────┴────┐
     ▼         ▼
   RULES       ML
     │         │
     └────┬────┘
          ▼
      RISK ENGINE
          │
     ┌────┴─────┐
     ▼          ▼
 DASHBOARD   REPORTING
```

------------------------------------------------------------------------

# 21. Development Order From Here

``` text
STAGE 1
Dataset Preparation
       │
       └── COMPLETED
              │
              ▼
STAGE 2
Canonical Internal Model
              │
              ▼
STAGE 3
Entity Resolution
              │
              ▼
STAGE 4
Correlation Engine
              │
              ▼
STAGE 5
Unified Timeline / Fusion
              │
              ▼
STAGE 6
Feature Engineering
              │
              ▼
STAGE 7
Rules + ML
              │
              ▼
STAGE 8
Risk Scoring
              │
       ┌──────┴──────┐
       ▼             ▼
STAGE 9          STAGE 10
Graph            API/Search
       │             │
       └──────┬──────┘
              ▼
STAGE 11
Dashboard
              │
              ▼
STAGE 12
Reporting
              │
              ▼
STAGE 13
Real PDF/Excel/Provider Ingestion
              │
              ▼
STAGE 14
Scale & Production Hardening
```

------------------------------------------------------------------------

# 22. Stage Gates

Each stage should be completed and validated before the project moves
forward.

## Gate 1 --- Dataset Readiness

**PASSED**

-   clean datasets available;
-   anomaly datasets available;
-   relationship ground truth available;
-   anomaly ground truth available;
-   anomaly population validated;
-   semantic links validated.

------------------------------------------------------------------------

## Gate 2 --- Canonical Model

Must prove all required Bank/CDR/IPDR fields can be represented in
stable internal schemas.

------------------------------------------------------------------------

## Gate 3 --- Entity Resolution

Must resolve cross-source identities without reading ground-truth
relationships as input.

------------------------------------------------------------------------

## Gate 4 --- Correlation

Must measure Bank-CDR and CDR-IPDR matching quality against ground
truth.

------------------------------------------------------------------------

## Gate 5 --- Fusion

Must correctly support:

``` text
0 matches
1 match
many matches
```

without losing Bank transactions.

------------------------------------------------------------------------

## Gate 6 --- Feature Engineering

Must pass:

-   temporal leakage tests;
-   reproducibility tests;
-   missing-context tests;
-   scenario-signal coverage checks.

------------------------------------------------------------------------

## Gate 7 --- Detection

Must outperform meaningful baselines using imbalance-aware metrics.

------------------------------------------------------------------------

## Gate 8 --- Risk Explanation

High-risk results must expose understandable evidence.

------------------------------------------------------------------------

## Gate 9 --- Graph Analytics

Must correctly reconstruct meaningful financial and communication
relationships.

------------------------------------------------------------------------

## Gates 10--14

Productisation stages should consume stable upstream interfaces rather
than reimplementing analytical logic.

------------------------------------------------------------------------

# 23. Current Progress Log

## Stage 1 --- COMPLETED

Completed work:

-   [x] Designed Bank dataset
-   [x] Designed CDR dataset
-   [x] Designed IPDR dataset
-   [x] Created clean Bank baseline
-   [x] Created clean CDR baseline
-   [x] Created clean IPDR baseline
-   [x] Created Bank-CDR semantic relationships
-   [x] Created CDR-IPDR semantic relationships
-   [x] Verified Bank -\> CDR -\> IPDR chains
-   [x] Created relationship ground truths
-   [x] Designed 15 suspicious scenario families
-   [x] Created Easy / Medium / Hard anomaly levels
-   [x] Created Bank-only anomalies
-   [x] Created Bank + CDR anomalies
-   [x] Created Bank + CDR + IPDR anomalies
-   [x] Reduced anomaly population to 100 controlled suspicious
    transactions
-   [x] Created anomaly ground truth
-   [x] Validated referential integrity
-   [x] Validated detectability
-   [x] Performed single-feature leakage audit
-   [x] Created clean implementation repository boundary

### Stage 1 Decision

**COMPLETE --- DATASETS FROZEN**

------------------------------------------------------------------------

# 24. Next Active Stage

## Stage 2 --- Canonical Internal Data Model

The immediate next work is:

1.  Define `BankTransaction`.
2.  Define `CDREvent`.
3.  Define `IPDRSession`.
4.  Define shared entity/identity representations.
5.  Define canonical timestamps.
6.  Define source provenance.
7.  Build loaders that map the current CSVs into these representations.
8.  Add schema validation tests.
9.  Verify that no information required for Stage 3/4 correlation is
    lost.

Only after Stage 2 passes its gate should implementation proceed to
entity resolution.

------------------------------------------------------------------------

# 25. Important Development Rule

Do **not** jump directly from prepared CSVs to model training.

The correct sequence is:

``` text
Prepared Data
     ↓
Canonical Model
     ↓
Entity Resolution
     ↓
Correlation
     ↓
Fusion / Timeline
     ↓
Features
     ↓
Rules + ML
```

The quality of the model depends heavily on the correctness of the
preceding stages.

------------------------------------------------------------------------

# 26. Core Experimental Question

One of TRI-NETRA's strongest quantitative experiments should be:

``` text
BANK ONLY
    │
    ▼
Detection performance


BANK + CDR
    │
    ▼
Detection performance


BANK + CDR + IPDR
    │
    ▼
Detection performance
```

Then compare:

``` text
Precision
Recall
F1
PR-AUC
ROC-AUC
False Positive Rate
Precision@K
Recall@K
```

This directly measures whether cross-domain data fusion adds
investigative value.

------------------------------------------------------------------------

# 27. Final Prototype Success Condition

The prototype should ultimately demonstrate:

``` text
Bank + CDR + IPDR
       ↓
Canonical representation
       ↓
Automatic entity matching
       ↓
Automatic event correlation
       ↓
Unified timeline
       ↓
Suspicious-pattern detection
       ↓
Risk-ranked transaction/entity
       ↓
Money + communication graph
       ↓
Investigator-readable evidence
       ↓
Forensic report
```

The final product should not merely answer:

> "Is this transaction anomalous?"

It should help answer:

> "What happened, when did it happen, which financial/telecom/network
> entities were involved, how are they connected, why is the activity
> suspicious, and what evidence supports further investigation?"

------------------------------------------------------------------------

# 28. Current Project Status

``` text
STAGE 1 — Dataset Preparation & Ground Truth      COMPLETE
STAGE 2 — Canonical Internal Data Model           COMPLETE
STAGE 3 — Entity Resolution                       COMPLETE
STAGE 4 — Correlation Engine                      NEXT
STAGE 5 — Unified Timeline / Fusion               PENDING
STAGE 6 — Feature Engineering                     PENDING
STAGE 7 — Rules + ML                              PENDING
STAGE 8 — Risk Scoring                            PENDING
STAGE 9 — Graph Analytics                         PENDING
STAGE 10 — Investigation API/Search               PENDING
STAGE 11 — Dashboard                              PENDING
STAGE 12 — Reporting                              PENDING
STAGE 13 — Multi-Format Ingestion                 FUTURE
STAGE 14 — Production Hardening                   FUTURE
```

**Current milestone:** Stage 1 is complete and frozen. Development
proceeds next with Stage 2.


## 9. Stage 3 — Entity Resolution (COMPLETE & FROZEN)
**Purpose**: Provides deterministic, typed identity resolution across Bank, CDR, and IPDR data to answer "Where does the same identity occur?" without creating false correlation assumptions.

**Input Contract**: Operates strictly on Stage 2 canonical objects (`BankTransaction`, `CDREvent`, `IPDRSession`).

**Identity Types**: Supports strict, closed-set mapping using `PHONE`, `CUSTOMER_ID`, `BANK_ACCOUNT`, `IMSI` (mobile subscriber/subscription identity), `IMEI` (mobile equipment/device identity), `CELL_ID`, and `IP_ADDRESS`.

**Identity Observations**: Preserves 1-to-many context mapping using lightweight `IdentityObservation` objects identifying source type, record ID, role, and timestamp.

**Registry Behavior**: `O(1)` memory-safe lookup mappings by combining identity type and normalized value. Prevents exact duplicate entries but maintains all distinct observations.

**Missing/Null**: Safely skips missing optional values, avoiding injection of dummy objects like `None` or `NaN` into the graph.

**Cross-Source Bridges**: Bank Phone ↔ CDR Phone (100% overlap match), CDR Phone ↔ IPDR MSISDN, CDR IMSI ↔ IPDR IMSI (56% overlap), CDR IMEI ↔ IPDR IMEI, CDR Cell ↔ IPDR Cell.

**Limitations**: Strict determinism. Semantic event correlation mapping belongs strictly to Stage 4. Behavior patterns are out of scope.

**Stage 4 Handoff**: Stage 4 can now query the `IdentityRegistry` for specific identities and retrieve their relevant temporal footprints without triggering full dataset scans.
