from enum import Enum
from pydantic import BaseModel
from datetime import datetime
from src.models.common import EntityIdentity, SourceType

class RoleType(str, Enum):
    SENDER = "SENDER"
    RECEIVER = "RECEIVER"
    A_PARTY = "A_PARTY"
    B_PARTY = "B_PARTY"
    SUBSCRIBER = "SUBSCRIBER"
    DEVICE = "DEVICE"
    CELL = "CELL"
    SOURCE_IP = "SOURCE_IP"
    DESTINATION_IP = "DESTINATION_IP"

class IdentityObservation(BaseModel):
    identity: EntityIdentity
    source_type: SourceType
    source_record_id: str
    source_field: str
    role: RoleType
    timestamp: datetime
