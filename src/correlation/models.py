from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from src.models.common import IdentityType, SourceType

class RelationshipType(str, Enum):
    BANK_CDR = "BANK_CDR"
    CDR_IPDR = "CDR_IPDR"

class MatchStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"

class IdentityMatchEvidence(BaseModel):
    identity_type: IdentityType
    normalized_value: str
    source_role: str
    target_role: str

class IdentityConflictEvidence(BaseModel):
    identity_type: IdentityType
    source_value: str
    target_value: str
    source_role: str
    target_role: str

class CorrelationRecord(BaseModel):
    correlation_id: str
    relationship_type: RelationshipType
    
    source_type: SourceType
    source_event_id: str
    source_timestamp: datetime
    
    target_type: SourceType
    target_event_id: str
    target_timestamp: datetime
    
    time_difference_seconds: int
    
    identity_evidence: List[IdentityMatchEvidence]
    conflicting_evidence: List[IdentityConflictEvidence]
    
    match_strength: MatchStrength
    accepted: bool
