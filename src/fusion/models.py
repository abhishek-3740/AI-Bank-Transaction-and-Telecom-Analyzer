from enum import Enum
from typing import List, Union
from datetime import datetime
from pydantic import BaseModel, Field

from src.models.bank import BankTransaction
from src.models.cdr import CDREvent
from src.models.ipdr import IPDRSession
from src.correlation.models import CorrelationRecord

class EventType(str, Enum):
    BANK = "BANK"
    CDR = "CDR"
    IPDR = "IPDR"

class TimelineEvent(BaseModel):
    event_type: EventType
    event_id: str
    timestamp: datetime
    event: Union[BankTransaction, CDREvent, IPDRSession]

class TransactionContext(BaseModel):
    transaction: BankTransaction
    cdr_events: List[CDREvent]
    ipdr_sessions: List[IPDRSession]
    bank_cdr_correlations: List[CorrelationRecord]
    cdr_ipdr_correlations: List[CorrelationRecord]
    timeline: List[TimelineEvent]
    
    @property
    def has_cdr_context(self) -> bool:
        return len(self.cdr_events) > 0
        
    @property
    def has_ipdr_context(self) -> bool:
        return len(self.ipdr_sessions) > 0
