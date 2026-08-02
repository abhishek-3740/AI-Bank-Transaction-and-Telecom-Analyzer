import re
import pandas as pd
from datetime import datetime
import ipaddress

def normalize_phone(phone) -> str:
    """
    Normalizes a phone number.
    - Converts to string.
    - Removes whitespace and hyphens.
    - Preserves digits only (removes leading '+').
    """
    if phone is None:
        return ""
    phone_str = str(phone).strip()
    if not phone_str or phone_str.lower() == 'nan' or phone_str == '<NA>':
        return ""
    # Keep digits only
    phone_str = re.sub(r'[^\d]', '', phone_str)
    return phone_str

def normalize_imsi(imsi) -> str:
    """Normalizes IMSI: convert to string, remove spaces."""
    if imsi is None:
        return ""
    imsi_str = str(imsi).strip()
    if not imsi_str or imsi_str.lower() == 'nan' or imsi_str == '<NA>':
        return ""
    return re.sub(r'\s+', '', imsi_str)

def normalize_imei(imei) -> str:
    """Normalizes IMEI: convert to string, remove spaces."""
    if imei is None:
        return ""
    imei_str = str(imei).strip()
    if not imei_str or imei_str.lower() == 'nan' or imei_str == '<NA>':
        return ""
    return re.sub(r'\s+', '', imei_str)

def parse_datetime(date_val, time_val) -> datetime:
    """
    Parses date and time into a single datetime object.
    Accepts string, pandas Timestamp, or handles missing.
    """
    date_str = str(date_val).strip() if not pd.isna(date_val) else ""
    time_str = str(time_val).strip() if not pd.isna(time_val) else ""
    
    if not date_str or date_str.lower() == 'nan':
        raise ValueError("Missing date for timestamp")
        
    if not time_str or time_str.lower() == 'nan':
        raise ValueError("Missing time for timestamp")
        
    # Standard format assumed: YYYY-MM-DD HH:MM:SS
    dt_str = f"{date_str} {time_str}"
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"Failed to parse datetime: {dt_str}") from e

def normalize_ip(ip) -> str:
    if ip is None or str(ip).strip().lower() == 'nan' or str(ip).strip() == '<NA>':
        return ""
    ip_str = str(ip).strip()
    if not ip_str:
        return ""
    try:
        ipaddress.ip_address(ip_str)
        return ip_str
    except ValueError:
        raise ValueError(f"Malformed IP address: {ip_str}")

def normalize_cell_id(cell_id) -> str:
    if cell_id is None or str(cell_id).strip().lower() == 'nan' or str(cell_id).strip() == '<NA>':
        return ""
    return str(cell_id).strip()

def normalize_account(account) -> str:
    if account is None or str(account).strip().lower() == 'nan' or str(account).strip() == '<NA>':
        return ""
    return str(account).strip()
