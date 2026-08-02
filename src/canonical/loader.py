import pandas as pd
from typing import Dict, Any, List, Tuple
from src.canonical.bank_mapper import map_bank_row
from src.canonical.cdr_mapper import map_cdr_row
from src.canonical.ipdr_mapper import map_ipdr_row

def load_bank(df: pd.DataFrame, file_path: str = "unknown") -> Tuple[List[Any], List[Dict[str, Any]], int]:
    """Returns (successful_records, errors, warning_count)."""
    successes = []
    errors = []
    warning_count = 0
    
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            record, warnings = map_bank_row(row_dict, file_path, i)
            successes.append(record)
            warning_count += len(warnings)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "row": row_dict})
            
    return successes, errors, warning_count

def load_cdr(df: pd.DataFrame, file_path: str = "unknown") -> Tuple[List[Any], List[Dict[str, Any]], int]:
    successes = []
    errors = []
    warning_count = 0
    
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            record, warnings = map_cdr_row(row_dict, file_path, i)
            successes.append(record)
            warning_count += len(warnings)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "row": row_dict})
            
    return successes, errors, warning_count

def load_ipdr(df: pd.DataFrame, file_path: str = "unknown") -> Tuple[List[Any], List[Dict[str, Any]], int]:
    successes = []
    errors = []
    warning_count = 0
    
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            record, warnings = map_ipdr_row(row_dict, file_path, i)
            successes.append(record)
            warning_count += len(warnings)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "row": row_dict})
            
    return successes, errors, warning_count
