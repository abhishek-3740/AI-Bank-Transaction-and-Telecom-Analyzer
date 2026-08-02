from decimal import Decimal
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from .common import SourceProvenance

class BankParty(BaseModel):
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    bank_name: Optional[str] = None
    account_number: Optional[str] = None
    account_type: Optional[str] = None
    ifsc: Optional[str] = None
    phone: Optional[str] = None

class BankTransaction(BaseModel):
    transaction_id: str
    transaction_timestamp: datetime
    transaction_reference: Optional[str] = None
    transaction_mode: Optional[str] = None
    currency: Optional[str] = None
    amount: Decimal
    
    sender: BankParty
    receiver: BankParty
    
    provenance: SourceProvenance
