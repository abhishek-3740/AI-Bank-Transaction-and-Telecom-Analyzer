from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from .common import SourceProvenance

class CDREvent(BaseModel):
    cdr_id: str
    event_timestamp: datetime
    a_party_phone: str
    b_party_phone: str
    call_type: Optional[str] = None
    duration_seconds: Optional[int] = None
    imsi: Optional[str] = None
    imei: Optional[str] = None
    first_bts_location: Optional[str] = None
    cell_id: Optional[str] = None
    roaming_circle: Optional[str] = None
    
    provenance: SourceProvenance
