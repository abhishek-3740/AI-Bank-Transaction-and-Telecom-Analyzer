import pandas as pd
from typing import Dict, Any, Tuple
from decimal import Decimal
from src.models.bank import BankTransaction, BankParty
from src.models.common import SourceProvenance, SourceType
from src.canonical.normalizers import parse_datetime, normalize_phone, normalize_account

def map_bank_row(row_dict: Dict[str, Any], file_path: str, row_index: int) -> Tuple[BankTransaction, list]:
    """
    Maps a raw bank CSV row dictionary to a BankTransaction canonical object.
    Returns (BankTransaction, warnings_list).
    Raises ValueError if core requirements fail.
    """
    warnings = []
    
    # 1. Provenance
    provenance = SourceProvenance(
        source_type=SourceType.BANK,
        source_file=file_path,
        source_record_id=str(row_dict.get('Transaction_ID', row_index))
    )
    
    # 2. Extract Core Fields
    txn_id = row_dict.get('Transaction_ID')
    if pd.isna(txn_id) or not txn_id:
        raise ValueError("Missing CORE_REQUIRED field: transaction_id")
    
    try:
        dt = parse_datetime(row_dict.get('Date'), row_dict.get('Timestamp'))
    except Exception as e:
        raise ValueError(f"Missing or invalid CORE_REQUIRED field: transaction_timestamp - {e}")
        
    try:
        amount_val = row_dict.get('Transaction_Amount')
        if pd.isna(amount_val):
            raise ValueError("Amount is null")
        amount = Decimal(str(amount_val))
    except Exception as e:
        raise ValueError(f"Missing or invalid CORE_REQUIRED field: amount - {e}")
        
    # 3. Extract Sender
    sender_phone = normalize_phone(row_dict.get('Sender_Phone_Number'))
    if not sender_phone:
        warnings.append("Missing FUSION_REQUIRED field: sender.phone")
        
    sender_account = normalize_account(row_dict.get('Sender_Account_Number'))
    
    sender = BankParty(
        customer_id=str(row_dict.get('Sender_Customer_ID')) if not pd.isna(row_dict.get('Sender_Customer_ID')) else None,
        customer_name=str(row_dict.get('Sender_Customer_Name')) if not pd.isna(row_dict.get('Sender_Customer_Name')) else None,
        bank_name=str(row_dict.get('Sender_Bank_Name')) if not pd.isna(row_dict.get('Sender_Bank_Name')) else None,
        account_number=sender_account if sender_account else None,
        account_type=str(row_dict.get('Sender_Account_Type')) if not pd.isna(row_dict.get('Sender_Account_Type')) else None,
        ifsc=str(row_dict.get('Sender_IFSC')) if not pd.isna(row_dict.get('Sender_IFSC')) else None,
        phone=sender_phone if sender_phone else None
    )
    
    # 4. Extract Receiver
    receiver_phone = normalize_phone(row_dict.get('Receiver_Phone_Number'))
    if not receiver_phone:
        warnings.append("Missing FUSION_REQUIRED field: receiver.phone")
        
    receiver_account = normalize_account(row_dict.get('Receiver_Account_Number'))

    receiver = BankParty(
        customer_id=str(row_dict.get('Receiver_Customer_ID')) if not pd.isna(row_dict.get('Receiver_Customer_ID')) else None,
        customer_name=str(row_dict.get('Receiver_Customer_Name')) if not pd.isna(row_dict.get('Receiver_Customer_Name')) else None,
        bank_name=str(row_dict.get('Receiver_Bank_Name')) if not pd.isna(row_dict.get('Receiver_Bank_Name')) else None,
        account_number=receiver_account if receiver_account else None,
        account_type=str(row_dict.get('Receiver_Account_Type')) if not pd.isna(row_dict.get('Receiver_Account_Type')) else None,
        ifsc=str(row_dict.get('Receiver_IFSC')) if not pd.isna(row_dict.get('Receiver_IFSC')) else None,
        phone=receiver_phone if receiver_phone else None
    )
    
    # 5. Build Canonical Object
    txn = BankTransaction(
        transaction_id=str(txn_id),
        transaction_timestamp=dt,
        transaction_reference=str(row_dict.get('Txn_Ref_Number')) if not pd.isna(row_dict.get('Txn_Ref_Number')) else None,
        transaction_mode=str(row_dict.get('Transaction_Mode')) if not pd.isna(row_dict.get('Transaction_Mode')) else None,
        currency=str(row_dict.get('Currency')) if not pd.isna(row_dict.get('Currency')) else None,
        amount=amount,
        sender=sender,
        receiver=receiver,
        provenance=provenance
    )
    
    return txn, warnings
