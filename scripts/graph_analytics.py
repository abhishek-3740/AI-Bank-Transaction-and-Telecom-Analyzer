#!/usr/bin/env python
"""Stage 9: graph / network analytics for TRI-NETRA.

Builds two graphs from the scored transactions and CDR data:

  1. Financial transfer graph — directed edge sender → receiver for every
     bank transaction. Node attributes: total_sent, total_received, out_degree,
     in_degree, in_out_ratio, max_risk_score, alert_count.

  2. Communication graph — directed edge A_party → B_party for every CDR
     VOICE call. Node attributes: total_calls, call_partners.

Then runs lightweight centrality analysis:
  - PageRank (damping=0.85) on the financial graph — high PR + high in_degree
    flags potential money laundering hubs.
  - In-degree / Out-degree ratio — mule accounts that only receive (never send)
    appear as ratio > 10 with in_degree > 1.

Writes:
  notebook/output/graph_analytics.csv   — node-level table (financial graph)
  notebook/output/graph_edges.csv       — edge-level table

Run: python scripts/graph_analytics.py
     (requires: python scripts/score.py to have run first)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "output"
DATA = ROOT / "data"

SCORED_CSV = OUT / "scored_transactions.csv"
BANK_CSV   = DATA / "anomalous" / "bank_anomaly.csv"
CDR_CSV    = DATA / "anomalous" / "cdr_anomaly.csv"
GT_CSV     = DATA / "ground_truth" / "anomaly_ground_truth.csv"

STR = {"Sender_Customer_ID": str, "Receiver_Account_Number": str,
       "Sender_Phone_Number": str, "Receiver_Phone_Number": str,
       "Sender_Account_Number": str, "Transaction_ID": str,
       "Receiver_Customer_ID": str}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not SCORED_CSV.exists():
        raise SystemExit(f"{SCORED_CSV} not found — run: python scripts/score.py")
    scored = pd.read_csv(SCORED_CSV, dtype={"Sender_Customer_ID": str,
                                             "Receiver_Account_Number": str,
                                             "Transaction_ID": str})
    bank = pd.read_csv(BANK_CSV, dtype=STR)
    cdr  = pd.read_csv(CDR_CSV,  dtype={"A_Party_Number": str, "B_Party_Number": str})
    gt   = pd.read_csv(GT_CSV,   dtype={"Customer_ID": str, "Transaction_ID": str})
    return scored, bank, cdr, gt


def _pagerank(adj: dict[str, dict[str, float]], damping: float = 0.85,
              max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
    """Simple power-iteration PageRank over a weighted directed adjacency dict."""
    nodes = list(adj.keys())
    # Ensure all destination nodes appear in the node list
    for src, dests in adj.items():
        for dst in dests:
            if dst not in adj:
                nodes.append(dst)
    nodes = list(dict.fromkeys(nodes))  # deduplicate, preserve order
    n = len(nodes)
    if n == 0:
        return {}
    idx = {node: i for i, node in enumerate(nodes)}
    rank = np.ones(n) / n
    # Out-degree weights
    out_w = np.zeros(n)
    for src, dests in adj.items():
        out_w[idx[src]] = sum(dests.values())

    for _ in range(max_iter):
        new_rank = np.ones(n) * (1 - damping) / n
        for src, dests in adj.items():
            s_i = idx[src]
            w_total = out_w[s_i]
            if w_total == 0:
                continue
            for dst, w in dests.items():
                new_rank[idx[dst]] += damping * rank[s_i] * (w / w_total)
        delta = np.abs(new_rank - rank).sum()
        rank = new_rank
        if delta < tol:
            break
    return {node: float(rank[i]) for i, node in enumerate(nodes)}


def build_financial_graph(bank: pd.DataFrame, scored: pd.DataFrame,
                           gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build node and edge tables for the sender→receiver financial graph."""
    # Merge risk scores onto bank rows (left join — bank has all rows)
    merged = bank.merge(
        scored[["Transaction_ID", "risk_score", "risk_band", "rules_fired",
                "is_suspicious_gt"]],
        on="Transaction_ID", how="left"
    )
    merged["risk_score"] = merged["risk_score"].fillna(0)
    merged["is_suspicious_gt"] = merged["is_suspicious_gt"].fillna(0).astype(int)

    # Edge table: one row per transaction
    edges = merged[["Transaction_ID", "Sender_Customer_ID", "Receiver_Account_Number",
                     "Transaction_Amount", "risk_score", "risk_band",
                     "is_suspicious_gt"]].copy()
    edges.rename(columns={"Sender_Customer_ID": "src",
                           "Receiver_Account_Number": "dst"}, inplace=True)

    # Build adjacency for PageRank (weight = transaction amount)
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in edges.iterrows():
        adj[row.src][row.dst] += float(row.Transaction_Amount)

    pr = _pagerank(dict(adj))

    # Node-level aggregates
    sent = edges.groupby("src").agg(
        total_sent=("Transaction_Amount", "sum"),
        out_degree=("dst", "count"),
        sent_alert_count=("is_suspicious_gt", "sum"),
        max_risk_sent=("risk_score", "max"),
    ).reset_index().rename(columns={"src": "node_id"})

    received = edges.groupby("dst").agg(
        total_received=("Transaction_Amount", "sum"),
        in_degree=("src", "count"),
        received_alert_count=("is_suspicious_gt", "sum"),
        max_risk_received=("risk_score", "max"),
    ).reset_index().rename(columns={"dst": "node_id"})

    nodes = sent.merge(received, on="node_id", how="outer").fillna(0)
    nodes["in_out_ratio"] = nodes["in_degree"] / (nodes["out_degree"] + 1e-6)
    nodes["total_amount"] = nodes["total_sent"] + nodes["total_received"]
    nodes["max_risk_score"] = nodes[["max_risk_sent", "max_risk_received"]].max(axis=1)
    nodes["alert_count"] = (nodes["sent_alert_count"] + nodes["received_alert_count"]).astype(int)
    nodes["pagerank"] = nodes["node_id"].map(pr).fillna(0)

    # Flag known mule accounts (appear only on receiver side in the GT)
    mule_ids = set(
        bank.loc[bank["Transaction_ID"].isin(set(gt.Transaction_ID)),
                 "Receiver_Account_Number"]
    )
    nodes["is_mule_account"] = nodes["node_id"].isin(mule_ids).astype(int)

    # Suspicion score: combines PageRank, in_out_ratio, alert_count
    nodes["suspicion_score"] = (
        0.4 * (nodes["pagerank"] / (nodes["pagerank"].max() + 1e-12)) +
        0.3 * np.clip(nodes["in_out_ratio"] / 10, 0, 1) +
        0.3 * (nodes["alert_count"] / (nodes["alert_count"].max() + 1e-12))
    ).round(4)

    nodes = nodes.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    edges = edges.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return nodes, edges


def build_communication_graph(cdr: pd.DataFrame) -> pd.DataFrame:
    """Aggregate node-level CDR statistics (call volume, unique partners)."""
    voice = cdr[cdr["Call_Type"] == "VOICE"].copy()
    stats_a = voice.groupby("A_Party_Number").agg(
        calls_made=("CDR_ID", "count"),
        unique_b_parties=("B_Party_Number", "nunique"),
        total_duration_s=("Call_Duration_Seconds", "sum"),
    ).reset_index().rename(columns={"A_Party_Number": "phone"})
    stats_b = voice.groupby("B_Party_Number").agg(
        calls_received=("CDR_ID", "count"),
        unique_a_parties=("A_Party_Number", "nunique"),
    ).reset_index().rename(columns={"B_Party_Number": "phone"})
    nodes = stats_a.merge(stats_b, on="phone", how="outer").fillna(0)
    nodes["total_calls"] = nodes["calls_made"] + nodes["calls_received"]
    nodes = nodes.sort_values("total_calls", ascending=False).reset_index(drop=True)
    return nodes


def main() -> None:
    print("loading data ...")
    scored, bank, cdr, gt = _load()

    print("building financial transfer graph ...")
    fin_nodes, fin_edges = build_financial_graph(bank, scored, gt)

    print("building communication graph ...")
    comm_nodes = build_communication_graph(cdr)

    OUT.mkdir(parents=True, exist_ok=True)
    fin_nodes.to_csv(OUT / "graph_analytics.csv", index=False)
    fin_edges.to_csv(OUT / "graph_edges.csv", index=False)
    comm_nodes.to_csv(OUT / "graph_cdr_nodes.csv", index=False)

    n_nodes = len(fin_nodes)
    n_edges = len(fin_edges)
    n_mules = fin_nodes["is_mule_account"].sum()
    top5 = fin_nodes.head(5)[["node_id", "in_degree", "out_degree",
                               "in_out_ratio", "pagerank", "suspicion_score",
                               "is_mule_account"]]
    print(f"\nfinancial graph: {n_nodes:,} nodes · {n_edges:,} edges")
    print(f"known mule accounts in top suspicious nodes: {n_mules}/{n_nodes}")
    print(f"\ntop 5 nodes by suspicion_score:")
    print(top5.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Validation: mule accounts should concentrate at the top
    top50_mules = fin_nodes.head(50)["is_mule_account"].sum()
    print(f"\nmule accounts in top-50 suspicious nodes: {top50_mules}/50 "
          f"({top50_mules / 50 * 100:.1f}%)")
    print(f"\nwrote {OUT / 'graph_analytics.csv'}")
    print(f"wrote {OUT / 'graph_edges.csv'}")
    print(f"wrote {OUT / 'graph_cdr_nodes.csv'}")


if __name__ == "__main__":
    main()
