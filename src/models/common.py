from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class SourceType(str, Enum):
    BANK = "BANK"
    CDR = "CDR"
    IPDR = "IPDR"
    UNKNOWN = "UNKNOWN"

class IdentityType(str, Enum):
    CUSTOMER_ID = "CUSTOMER_ID"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    PHONE = "PHONE"
    MSISDN = "MSISDN"
    IMSI = "IMSI"
    IMEI = "IMEI"
    IP_ADDRESS = "IP_ADDRESS"
    CELL_ID = "CELL_ID"

class SourceProvenance(BaseModel):
    source_type: SourceType
    source_file: str
    source_record_id: str

class EntityIdentity(BaseModel):
    identity_type: IdentityType
    raw_value: str
    normalized_value: str
