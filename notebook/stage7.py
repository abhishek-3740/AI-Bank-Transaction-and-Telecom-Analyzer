import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List

from sklearn import datasets
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    precision_recall_curve, auc, classification_report
)

# Enforce strict reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Environment setup complete.")

# File paths
PATH_A = "C:\\Users\\Arpit Mishra\\Desktop\\AI-Bank-Transaction-and-Telecom-Analyzer\\data\\clean\\bank_final.csv"  # Replace with exact Stage 6 output paths
PATH_B = "C:\\Users\\Arpit Mishra\\Desktop\\AI-Bank-Transaction-and-Telecom-Analyzer\\data\\clean\\cdr_final.csv"
PATH_C = "C:\\Users\\Arpit Mishra\\Desktop\\AI-Bank-Transaction-and-Telecom-Analyzer\\data\\clean\\ipdr_final.csv"
PATH_GT = "C:\\Users\\Arpit Mishra\\Desktop\\AI-Bank-Transaction-and-Telecom-Analyzer\\data\\ground_truth\\anomaly_ground_truth.csv"

def load_and_label(path_a: str, path_b: str, path_c: str, path_gt: str):
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    df_c = pd.read_csv(path_c)
    gt_df = pd.read_csv(path_gt)

    # Validate row alignment across sets A, B, and C
    assert len(df_a) == len(df_b) == len(df_c), "Row count mismatch across feature sets!"
    assert (df_a['transaction_id'] == df_b['transaction_id']).all(), "Transaction ID mismatch between A and B!"

    # Identify anomalous transaction IDs from ground truth
    anomalous_ids = set(gt_df['transaction_id'].unique())
    
    for df in [df_a, df_b, df_c]:
        df['target'] = df['transaction_id'].apply(lambda tx_id: 1 if tx_id in anomalous_ids else 0)

    print(f"Loaded {len(df_a)} transaction contexts.")
    print(f"Total Anomalies: {df_a['target'].sum()} ({df_a['target'].mean()*100:.2f}% anomaly prevalence)")
    return df_a, df_b, df_c

# Un-comment when Stage 6 feature files are ready:
df_a, df_b, df_c = load_and_label(PATH_A, PATH_B, PATH_C, PATH_GT)


def temporal_split(df: pd.DataFrame, time_col: str = 'transaction_timestamp', train_ratio=0.70, val_ratio=0.15):
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()
    
    return train, val, test

def evaluate_model(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5, k_values: List[int] = [50, 100, 250]):
    y_pred = (y_scores >= threshold).astype(int)
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(r_curve, p_curve)
    
    metrics = {'Precision': p, 'Recall': r, 'F1': f1, 'PR-AUC': pr_auc}
    
    # Calculate Precision@K and Recall@K
    top_k_idx = np.argsort(y_scores)[::-1]
    total_positives = np.sum(y_true)
    
    for k in k_values:
        hits = np.sum(y_true[top_k_idx[:k]])
        metrics[f'P@{k}'] = hits / k
        metrics[f'R@{k}'] = hits / total_positives if total_positives > 0 else 0.0
        
    return metrics

def run_ablation_experiments(datasets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]):
    exclude_cols = ['transaction_id', 'transaction_timestamp', 'target', 'customer_id']
    results = []

    for name, (train_df, val_df, test_df) in datasets.items():
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        # Fit missing value imputation strictly on TRAIN set
        medians = train_df[feature_cols].median()
        X_train = train_df[feature_cols].fillna(medians).values
        X_val = val_df[feature_cols].fillna(medians).values
        X_test = test_df[feature_cols].fillna(medians).values
        
        y_train, y_val, y_test = train_df['target'].values, val_df['target'].values, test_df['target'].values

        # --- 1. Isolation Forest (Unsupervised) ---
        iso = IsolationForest(n_estimators=100, random_state=RANDOM_SEED, contamination='auto')
        iso.fit(X_train)
        iso_scores = -iso.decision_function(X_test)  # Higher = More anomalous
        iso_res = evaluate_model(y_test, iso_scores, threshold=np.percentile(iso_scores, 95))
        iso_res.update({'Model': 'Isolation Forest', 'Feature Set': name})
        results.append(iso_res)

        # --- 2. Random Forest (Supervised Baseline) ---
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Threshold tuning on Validation Set
        rf_val_scores = rf.predict_proba(X_val)[:, 1]
        p_val, r_val, threshs = precision_recall_curve(y_val, rf_val_scores)
        f1s = 2 * (p_val * r_val) / (p_val + r_val + 1e-10)
        best_thresh = threshs[np.argmax(f1s)] if len(threshs) > 0 else 0.5
        
        rf_test_scores = rf.predict_proba(X_test)[:, 1]
        rf_res = evaluate_model(y_test, rf_test_scores, threshold=best_thresh)
        rf_res.update({'Model': 'Random Forest', 'Feature Set': name})
        results.append(rf_res)

        # --- 3. XGBoost (Gradient Boosting) ---
        scale_pos = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
        xgb = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos, random_state=RANDOM_SEED, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        
        xgb_test_scores = xgb.predict_proba(X_test)[:, 1]
        xgb_res = evaluate_model(y_test, xgb_test_scores, threshold=best_thresh)
        xgb_res.update({'Model': 'XGBoost', 'Feature Set': name})
        results.append(xgb_res)

    # Master Ablation Table
    summary_df = pd.DataFrame(results)
    cols = ['Model', 'Feature Set', 'Precision', 'Recall', 'F1', 'PR-AUC', 'P@100', 'R@100']
    return summary_df[cols]

# Create the datasets dictionary containing Train, Val, and Test splits for A, B, and C
datasets = {
    'Set A (Bank)': (train_a, val_a, test_a),
    'Set B (Bank + CDR)': (train_b, val_b, test_b),
    'Set C (Bank + CDR + IPDR)': (train_c, val_c, test_c)
}