from pydantic import BaseModel
from typing import Optional

class FeatureRow(BaseModel):
    transaction_id: str
    
    # BANK FEATURES
    transaction_amount: Optional[float] = None
    transaction_hour: Optional[int] = None
    customer_history_count: int = 0
    
    amount_vs_customer_median: Optional[float] = None
    amount_robust_zscore: Optional[float] = None
    amount_percentile: Optional[float] = None
    
    receiver_seen_before: int = 0
    receiver_historical_count: int = 0
    receiver_frequency: float = 0.0
    
    hour_rarity: Optional[float] = None
    
    txn_count_previous_10m: int = 0
    txn_count_previous_30m: int = 0
    txn_count_previous_1h: int = 0
    
    amount_velocity_30m: float = 0.0
    amount_velocity_1h: float = 0.0
    
    time_since_previous_transaction: Optional[float] = None
    
    # CDR FEATURES
    has_cdr_context: int = 0
    
    calls_previous_10m: int = 0
    calls_previous_30m: int = 0
    calls_previous_1h: int = 0
    
    nearest_call_before_seconds: Optional[float] = None
    
    total_call_duration_30m: int = 0
    max_call_duration_30m: Optional[float] = None
    
    caller_novelty: Optional[int] = None
    caller_historical_frequency: Optional[float] = None
    
    imei_novelty: Optional[int] = None
    cell_novelty: Optional[int] = None
    roaming_change: Optional[int] = None
    
    # IPDR FEATURES
    has_ipdr_context: int = 0
    
    sessions_previous_10m: int = 0
    sessions_previous_30m: int = 0
    
    nearest_session_before_seconds: Optional[float] = None
    
    source_ip_novelty: Optional[int] = None
    destination_ip_novelty: Optional[int] = None
    destination_port_novelty: Optional[int] = None
    
    imsi_imei_pair_novelty: Optional[int] = None
    
    device_consistency: Optional[int] = None
    cell_consistency: Optional[int] = None
    
    session_duration_deviation: Optional[float] = None
