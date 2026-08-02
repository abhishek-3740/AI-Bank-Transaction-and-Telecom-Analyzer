from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from .common import SourceProvenance

class IPDRSession(BaseModel):
    ipdr_id: str
    session_timestamp: datetime
    subscriber_imsi: Optional[str] = None
    subscriber_msisdn: str
    device_imei: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    destination_port: Optional[int] = None
    cell_id: Optional[str] = None
    duration_seconds: Optional[int] = None
    
    provenance: SourceProvenance
