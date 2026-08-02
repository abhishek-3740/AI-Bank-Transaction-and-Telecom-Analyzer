from pydantic import BaseModel

class CorrelationConfig(BaseModel):
    bank_cdr_window_seconds: int = 1800
    cdr_ipdr_window_seconds: int = 1800
