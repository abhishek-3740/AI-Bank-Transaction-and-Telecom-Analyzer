import pandas as pd
from typing import Dict, Any, Tuple
from src.models.ipdr import IPDRSession
from src.models.common import SourceProvenance, SourceType
from src.canonical.normalizers import parse_datetime, normalize_phone, normalize_imsi, normalize_imei, normalize_cell_id, normalize_ip

def map_ipdr_row(row_dict: Dict[str, Any], file_path: str, row_index: int) -> Tuple[IPDRSession, list]:
    warnings = []
    
    provenance = SourceProvenance(
        source_type=SourceType.IPDR,
        source_file=file_path,
        source_record_id=str(row_dict.get('IPDR_ID', row_index))
    )
    
    ipdr_id = row_dict.get('IPDR_ID')
    if pd.isna(ipdr_id) or not ipdr_id:
        raise ValueError("Missing CORE_REQUIRED field: ipdr_id")
        
    try:
        dt = parse_datetime(row_dict.get('Session_Date'), row_dict.get('Session_Start_Time'))
    except Exception as e:
        raise ValueError(f"Missing or invalid CORE_REQUIRED field: session_timestamp - {e}")
        
    msisdn = normalize_phone(row_dict.get('Subscriber_MSISDN'))
    if not msisdn:
        raise ValueError("Missing CORE_REQUIRED field: subscriber_msisdn")
        
    imsi = normalize_imsi(row_dict.get('Subscriber_IMSI'))
    if not imsi:
        warnings.append("Missing FUSION_REQUIRED field: subscriber_imsi")
        
    imei = normalize_imei(row_dict.get('Device_IMEI'))
    if not imei:
        warnings.append("Missing FUSION_REQUIRED field: device_imei")
        
    cell_id = normalize_cell_id(row_dict.get('Cell_Global_ID'))
    if not cell_id:
        warnings.append("Missing FUSION_REQUIRED field: cell_id")
        
    duration = None
    if not pd.isna(row_dict.get('Session_Duration_Seconds')):
        try:
            duration = int(row_dict.get('Session_Duration_Seconds'))
            if duration < 0:
                raise ValueError("Duration cannot be negative")
        except ValueError as e:
            raise ValueError(f"Invalid Session_Duration_Seconds: {e}")
            
    port = None
    if not pd.isna(row_dict.get('Destination_Port')):
        try:
            port = int(row_dict.get('Destination_Port'))
            if not (0 <= port <= 65535):
                raise ValueError("Port must be between 0 and 65535")
        except ValueError as e:
            raise ValueError(f"Invalid Destination_Port: {e}")
            
    try:
        src_ip = normalize_ip(row_dict.get('Source_IP_Address'))
    except ValueError as e:
        raise ValueError(f"Invalid Source_IP_Address: {e}")
        
    try:
        dst_ip = normalize_ip(row_dict.get('Destination_IP_Address'))
    except ValueError as e:
        raise ValueError(f"Invalid Destination_IP_Address: {e}")

    session = IPDRSession(
        ipdr_id=str(ipdr_id),
        session_timestamp=dt,
        subscriber_msisdn=msisdn,
        subscriber_imsi=imsi if imsi else None,
        device_imei=imei if imei else None,
        source_ip=src_ip if src_ip else None,
        destination_ip=dst_ip if dst_ip else None,
        destination_port=port,
        cell_id=cell_id if cell_id else None,
        duration_seconds=duration,
        provenance=provenance
    )
    
    return session, warnings
