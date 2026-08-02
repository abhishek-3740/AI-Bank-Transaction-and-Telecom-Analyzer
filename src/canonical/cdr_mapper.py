import pandas as pd
from typing import Dict, Any, Tuple
from src.models.cdr import CDREvent
from src.models.common import SourceProvenance, SourceType
from src.canonical.normalizers import parse_datetime, normalize_phone, normalize_imsi, normalize_imei, normalize_cell_id

def map_cdr_row(row_dict: Dict[str, Any], file_path: str, row_index: int) -> Tuple[CDREvent, list]:
    warnings = []
    
    provenance = SourceProvenance(
        source_type=SourceType.CDR,
        source_file=file_path,
        source_record_id=str(row_dict.get('CDR_ID', row_index))
    )
    
    cdr_id = row_dict.get('CDR_ID')
    if pd.isna(cdr_id) or not cdr_id:
        raise ValueError("Missing CORE_REQUIRED field: cdr_id")
        
    try:
        dt = parse_datetime(row_dict.get('Call_Date'), row_dict.get('Call_Start_Time'))
    except Exception as e:
        raise ValueError(f"Missing or invalid CORE_REQUIRED field: event_timestamp - {e}")
        
    a_party = normalize_phone(row_dict.get('A_Party_Number'))
    if not a_party:
        raise ValueError("Missing CORE_REQUIRED field: a_party_phone")
        
    b_party = normalize_phone(row_dict.get('B_Party_Number'))
    if not b_party:
        raise ValueError("Missing CORE_REQUIRED field: b_party_phone")
        
    imsi = normalize_imsi(row_dict.get('IMSI'))
    if not imsi:
        warnings.append("Missing FUSION_REQUIRED field: imsi")
        
    imei = normalize_imei(row_dict.get('IMEI'))
    if not imei:
        warnings.append("Missing FUSION_REQUIRED field: imei")
        
    cell_id = normalize_cell_id(row_dict.get('First_Cell_Global_ID'))
    if not cell_id:
        warnings.append("Missing FUSION_REQUIRED field: cell_id")
        
    duration = None
    if not pd.isna(row_dict.get('Call_Duration_Seconds')):
        try:
            duration = int(row_dict.get('Call_Duration_Seconds'))
            if duration < 0:
                raise ValueError("Duration cannot be negative")
        except ValueError as e:
            raise ValueError(f"Invalid Call_Duration_Seconds: {e}")

    event = CDREvent(
        cdr_id=str(cdr_id),
        event_timestamp=dt,
        a_party_phone=a_party,
        b_party_phone=b_party,
        call_type=str(row_dict.get('Call_Type')) if not pd.isna(row_dict.get('Call_Type')) else None,
        duration_seconds=duration,
        imsi=imsi if imsi else None,
        imei=imei if imei else None,
        first_bts_location=str(row_dict.get('First_BTS_Location')) if not pd.isna(row_dict.get('First_BTS_Location')) else None,
        cell_id=cell_id if cell_id else None,
        roaming_circle=str(row_dict.get('Roaming_Network_Circle')) if not pd.isna(row_dict.get('Roaming_Network_Circle')) else None,
        provenance=provenance
    )
    
    return event, warnings
