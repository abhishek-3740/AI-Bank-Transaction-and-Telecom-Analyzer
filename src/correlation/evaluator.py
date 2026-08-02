import csv
from typing import List, Dict, Any
from src.correlation.models import CorrelationRecord

def evaluate_predictions(
    predictions: List[CorrelationRecord],
    ground_truth_path: str,
    source_id_col: str,
    target_id_col: str
) -> Dict[str, Any]:
    
    truth_set = set()
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Is_Correlated") == "1":
                    truth_set.add((row[source_id_col], row[target_id_col]))
    except FileNotFoundError:
        pass # Allow running without ground truth safely
                
    pred_set = set()
    pred_map = {}
    for p in predictions:
        if p.accepted:
            key = (p.source_event_id, p.target_event_id)
            pred_set.add(key)
            pred_map[key] = p
            
    if not truth_set:
        return {
            "truth_links": 0,
            "predicted_links": len(pred_set),
            "TP": 0, "FP": 0, "FN": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "fp_samples": [], "fn_samples": []
        }
            
    tp_set = pred_set.intersection(truth_set)
    fp_set = pred_set - truth_set
    fn_set = truth_set - pred_set
    
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    fp_samples = []
    for key in list(fp_set)[:5]:
        p = pred_map[key]
        fp_samples.append({
            "source_id": key[0],
            "target_id": key[1],
            "time_difference": p.time_difference_seconds,
            "evidence": [{"type": e.identity_type.value, "val": e.normalized_value} for e in p.identity_evidence],
            "conflicts": [{"type": c.identity_type.value, "s": c.source_value, "t": c.target_value} for c in p.conflicting_evidence],
            "strength": p.match_strength.value
        })
        
    fn_samples = []
    for key in list(fn_set)[:5]:
        fn_samples.append({
            "source_id": key[0],
            "target_id": key[1]
        })
        
    return {
        "truth_links": len(truth_set),
        "predicted_links": len(pred_set),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_samples": fp_samples,
        "fn_samples": fn_samples
    }
