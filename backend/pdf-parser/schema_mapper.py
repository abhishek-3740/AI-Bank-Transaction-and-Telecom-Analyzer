"""
schema_mapper.py

Handles canonical schema definitions, refined alias dictionaries, 
weighted dataset detection utilizing fuzzy scoring, and a highly optimized 
4-tier semantic mapping system with cached embeddings and static data.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Set, Tuple
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class UnknownDatasetError(Exception):
    pass

class AmbiguousDatasetError(Exception):
    pass

# ==========================================
# CANONICAL SCHEMAS
# ==========================================

BANK_SCHEMA = [
    "Transaction_ID", "Date", "Timestamp", "Txn_Ref_Number", "Transaction_Mode", 
    "Currency", "Transaction_Amount", "Sender_Customer_ID", "Sender_Customer_Name", 
    "Sender_Bank_Name", "Sender_Account_Number", "Sender_Account_Type", "Sender_IFSC", 
    "Sender_Phone_Number", "Receiver_Customer_ID", "Receiver_Customer_Name", 
    "Receiver_Bank_Name", "Receiver_Account_Number", "Receiver_Account_Type", 
    "Receiver_IFSC", "Receiver_Phone_Number"
]

CDR_SCHEMA = [
    "CDR_ID", "Call_Date", "Call_Start_Time", "A_Party_Number", "B_Party_Number", 
    "Call_Type", "Call_Duration_Seconds", "IMSI", "IMEI", "First_BTS_Location", 
    "First_Cell_Global_ID", "Roaming_Network_Circle"
]

IPDR_SCHEMA = [
    "IPDR_ID", "Session_Date", "Session_Start_Time", "Subscriber_IMSI", "Subscriber_MSISDN", 
    "Device_IMEI", "Source_IP_Address", "Destination_IP_Address", "Destination_Port", 
    "Cell_Global_ID", "Session_Duration_Seconds"
]

SCHEMAS = {
    "bank": BANK_SCHEMA,
    "cdr": CDR_SCHEMA,
    "ipdr": IPDR_SCHEMA
}

# ==========================================
# EXPANDED AND REFINED ALIAS DICTIONARIES
# ==========================================

BANK_ALIASES = {
    "Date": [
        "Txn Date", "Transaction Date", "Posting Date", "Value Date", "Date",
        "Booking Date", "Tran Date", "Tran Date", "Transaction Date", "Value Dt",
        "Tran Dt", "Txn Dt", "Dt"
    ],
    "Txn_Ref_Number": [
        "UTR Number", "Reference", "Ref No", "Reference Number", "Reference No.",
        "Cheque/Ref No.", "Transaction ID", "Transaction Reference", "UTR",
        "Cheque Number", "Chq No", "Chq No.", "Cheque No", "Cheque No.",
        "Ref. No.", "Ref No.", "Txn Ref", "Transaction Ref", "Chq.No",
        "Chq/Ref No", "Chq / Ref No", "Reference No", "Cheque/Ref.No.",
        "Chq.No.", "Chq/Ref", "Chq/Ref.", "UTR No", "UTRNo"
    ],
    "Transaction_Amount": [
        "Amount", "Txn Amount", "Transaction Amount", "Transfer Amount",
        "Transaction_Amount_Merged", "Amount(INR)", "Amount (INR)",
        "Amt", "Amt (INR)", "Amount INR", "Txn Amt", "Debit Amount",
        "Credit Amount", "Withdrawal Amt", "Deposit Amt", "Amount (Rs.)",
        "Amount(Rs.)", "Amt.", "Txn Amt.", "Amount (INR)", "Amt(INR)",
        "Amt (INR)", "Amount INR", "Amount in INR", "INR Amount",
        "Rs.", "Rs", "Rupees", "Amount (Rs)", "Amt (Rs)", "Amt(Rs)",
        "Withdrawal", "Deposit", "Debit", "Credit", "Dr Amount", "Cr Amount",
        "Transaction Amount Merged", "Amount Merged", "Merged Amount"
    ],
    "Sender_Account_Number": [
        "Account No", "A/C No", "Sender A/C", "Account Number", "Account",
        "Debit Account", "Source Account", "A/c No.", "Acct No", "Acct Number",
        "A/C Number", "Acc No", "Acc Number", "Account No.", "Acct No.",
        "Sender Account", "Sender Account Number", "From Account"
    ],
    "Receiver_Account_Number": [
        "Beneficiary Account", "Receiver Account", "Destination Account",
        "Credit Account", "Beneficiary A/C", "Receiver A/C",
        "Beneficiary A/C No", "Receiver Account Number", "To Account",
        "Credit A/C", "Beneficiary Account Number", "Receiver Acc No"
    ],
    "Transaction_Mode": [
        "Mode", "Type", "Particulars", "Description", "Remarks", "Narration",
        "Transaction Details", "Details", "Transaction Particulars",
        "Txn Particulars", "Particulars", "Narration", "Description",
        "Transaction Description", "Txn Description", "Transaction Particulars",
        "Txn Details", "Transaction Narration", "Txn Narration", "Particulars",
        "Transaction Type", "Txn Type", "Mode of Transaction", "Payment Mode"
    ],
    "Sender_Phone_Number": [
        "Mobile", "Phone", "Contact", "Phone Number", "Mobile No", "Mobile Number",
        "Contact Number", "Phone No.", "Sender Mobile", "Sender Phone",
        "Sender Contact", "From Mobile", "From Phone"
    ],
    "Sender_IFSC": [
        "IFSC", "IFSC Code", "Sender IFSC", "IFSC_Code", "Ifsc Code",
        "IFSC_Code", "Sender IFSC Code", "From IFSC", "IFSC Code:"
    ],
    "Transaction_ID": [
        "Transaction ID", "Txn ID", "Transaction No", "Txn No", "Txn Number",
        "Transaction Number", "Voucher No", "Voucher Number", "Txn ID",
        "Transaction Ref No", "Ref ID", "Transaction Ref", "Txn Ref",
        "Reference ID", "Transaction Reference Number", "Txn Ref No"
    ],
    "Timestamp": [
        "Time", "Txn Time", "Transaction Time", "Time Stamp", "Timestamp",
        "Entry Time", "Posting Time", "Transaction Timestamp", "Txn Timestamp",
        "Value Time", "Tran Time"
    ],
    "Currency": [
        "Curr", "Currency", "CCY", "Cur", "Currency Code", "Currency Type",
        "Txn Currency", "Transaction Currency"
    ],
    "Sender_Customer_Name": [
        "Customer Name", "Sender Name", "Account Holder", "Holder Name",
        "Customer", "Sender", "From Name", "Remitter Name", "Account Holder Name",
        "Sender's Name", "Remitter", "Drawer"
    ],
    "Sender_Bank_Name": [
        "Bank", "Bank Name", "Sender Bank", "From Bank", "Remitter Bank",
        "Branch Name", "Branch", "Bank/Branch", "Sender Bank Name",
        "Remitting Bank", "Drawee Bank", "Bank/Branch Name"
    ],
    "Sender_Account_Type": [
        "Account Type", "A/C Type", "Acct Type", "Type of Account", "Scheme",
        "Sender Account Type", "From Account Type", "Account Category"
    ],
    "Receiver_Customer_Name": [
        "Beneficiary Name", "Receiver Name", "To Name", "Beneficiary",
        "Payee Name", "Payee", "Receiver's Name", "Beneficiary's Name",
        "Creditor Name", "Recipient Name"
    ],
    "Receiver_Bank_Name": [
        "Beneficiary Bank", "To Bank", "Receiver Bank", "Payee Bank",
        "Beneficiary Bank Name", "Receiving Bank", "Credit Bank",
        "Recipient Bank", "Destination Bank"
    ],
    "Receiver_Account_Type": [
        "Beneficiary Account Type", "Receiver A/C Type", "To Account Type",
        "Receiver Account Type", "Beneficiary A/C Type", "Credit Account Type"
    ],
    "Receiver_IFSC": [
        "Beneficiary IFSC", "Receiver IFSC", "To IFSC", "Beneficiary IFSC Code",
        "Receiver IFSC Code", "Payee IFSC", "Credit IFSC", "Destination IFSC"
    ],
    "Receiver_Phone_Number": [
        "Beneficiary Mobile", "Receiver Mobile", "To Mobile", "Beneficiary Phone",
        "Receiver Phone", "Payee Phone", "Receiver Phone Number",
        "Beneficiary Phone Number", "Payee Mobile", "Recipient Phone"
    ],
    "Sender_Customer_ID": [
        "Customer ID", "Cust ID", "Sender ID", "Client ID", "CIF", "CIF No",
        "Customer Identification", "Sender Customer ID", "From Customer ID",
        "CIF Number", "Cust ID No"
    ],
    "Receiver_Customer_ID": [
        "Beneficiary ID", "Receiver ID", "To Customer ID", "Beneficiary Cust ID",
        "Receiver Customer ID", "Payee ID", "Beneficiary Customer ID",
        "Recipient ID", "Creditor ID"
    ],
}

CDR_ALIASES = {
    "Call_Date": ["Date", "Call Date", "Start Date", "Call Dt", "Date of Call", "Call Date"],
    "Call_Start_Time": ["Time", "Start Time", "Call Time", "Call Start Time", "Time of Call", "Call Time"],
    "A_Party_Number": [
        "Calling Number", "Originating Number", "Caller", "A Party", "Calling No",
        "A Number", "Calling Party", "Origin Number", "From Number", "A-Party",
        "Calling No.", "A Party Number", "Origin No"
    ],
    "B_Party_Number": [
        "Called Number", "Destination Number", "Receiver", "B Party", "Called No",
        "B Number", "Called Party", "Destination Party", "To Number", "B-Party",
        "Called No.", "B Party Number", "Destination No", "Recipient Number"
    ],
    "Call_Duration_Seconds": [
        "Duration", "Call Duration", "Time (sec)", "Duration (s)", "Secs",
        "Total Duration", "Call Duration (sec)", "Duration Sec", "Dur (sec)",
        "Duration in Sec", "Call Duration (s)", "Duration (Seconds)", "Duration Seconds"
    ],
    "First_BTS_Location": [
        "Location", "Tower Location", "BTS", "Site ID", "Cell Site", "Address",
        "BTS Location", "Tower Address", "Site Address", "BTS Address",
        "Cell Tower Location", "First BTS Location", "Tower ID"
    ],
    "Call_Type": [
        "Type", "Call Type", "Voice/SMS", "Service Type", "Call Nature",
        "Communication Type", "Direction", "Call Direction", "Service",
        "Call Service Type", "Communication Mode"
    ],
    "IMSI": ["IMSI Number", "Subscriber IMSI", "IMSI No", "IMSI Code", "IMSI ID"],
    "IMEI": ["IMEI Number", "Handset IMEI", "Device IMEI", "IMEI No", "IMEI Code", "IMEI ID"],
    "First_Cell_Global_ID": [
        "Cell ID", "CGI", "Cell Global ID", "Global Cell ID", "Cell Identifier",
        "Cell Global Identifier", "First Cell ID", "CGI Code", "Cell ID Code"
    ],
    "Roaming_Network_Circle": [
        "Roaming Circle", "Network Circle", "Circle", "Roaming Network", "Network",
        "Roaming Zone", "Network Zone", "Circle Name", "Roaming Area"
    ],
    "CDR_ID": [
        "CDR ID", "Record ID", "Call Record ID", "Record Number", "CDR Number",
        "Call Record Number", "CDR Record ID", "Record ID Number"
    ],
}

IPDR_ALIASES = {
    "Session_Date": [
        "Date", "Session Date", "Start Date", "Session Dt", "Date of Session",
        "Session Start Date", "IPDR Date"
    ],
    "Session_Start_Time": [
        "Time", "Start Time", "Session Time", "Session Start Time", "Time of Session",
        "Session Timestamp", "IPDR Time", "Session Start Timestamp"
    ],
    "Source_IP_Address": [
        "Src IP", "Source IP", "Origin IP", "Private IP", "Framed IP",
        "Source IP Address", "Client IP", "User IP", "Originating IP",
        "Source IP Addr", "Src IP Address", "Local IP"
    ],
    "Destination_IP_Address": [
        "Dest IP", "Destination IP", "Target IP", "Server IP", "Destination IP Address",
        "Dst IP", "Remote IP", "Destination IP Addr", "Dest IP Address",
        "Target IP Address", "External IP", "Server IP Address"
    ],
    "Destination_Port": [
        "Dest Port", "Target Port", "Port", "Server Port", "Destination Port",
        "Dst Port", "Service Port", "Target Port Number", "Destination Port Number",
        "Port Number", "Server Port Number"
    ],
    "Session_Duration_Seconds": [
        "Duration", "Time (sec)", "Uptime", "Session Duration", "Session Duration (sec)",
        "Duration Sec", "Dur (sec)", "Session Duration (s)", "Duration Seconds",
        "Session Time", "Connection Duration", "Session Length"
    ],
    "Subscriber_MSISDN": [
        "MSISDN", "Mobile Number", "Phone Number", "Subscriber Number", "Mobile No",
        "Cellular Number", "MSISDN Number", "Subscriber Mobile", "Subscriber Phone",
        "MSISDN No", "Mobile Number", "Cell Number"
    ],
    "Subscriber_IMSI": [
        "IMSI", "Subscriber IMSI", "IMSI Number", "IMSI No", "Subscriber IMSI Number",
        "IMSI Code", "IMSI ID", "Subscriber IMSI Code"
    ],
    "Device_IMEI": [
        "IMEI", "Device IMEI", "IMEI Number", "Handset IMEI", "IMEI No",
        "Device IMEI Number", "IMEI Code", "IMEI ID", "Handset IMEI Number"
    ],
    "Cell_Global_ID": [
        "Cell ID", "CGI", "Cell Global ID", "Global Cell ID", "Cell Identifier",
        "Cell Global Identifier", "CGI Code", "Cell ID Code", "Location ID",
        "Cell Location ID", "Global Cell Identifier"
    ],
    "IPDR_ID": [
        "IPDR ID", "Record ID", "Session Record ID", "Record Number", "IPDR Number",
        "IPDR Record ID", "Session ID", "Record ID Number", "IPDR Record Number"
    ],
}

ALIASES = {
    "bank": BANK_ALIASES,
    "cdr": CDR_ALIASES,
    "ipdr": IPDR_ALIASES
}

# ==========================================
# STATIC CACHE & WEIGHTS
# ==========================================

HIGH_WEIGHT_TERMS = {
    "txn_ref_number", "transaction_id", "sender_account_number", "sender_ifsc", "receiver_account_number",
    "utr number", "reference", "utr", "ifsc code",
    "a_party_number", "b_party_number", "imsi", "imei", "first_cell_global_id",
    "source_ip_address", "destination_ip_address", "destination_port", "subscriber_msisdn"
}

_dataset_terms_cache = {}
_all_valid_terms = set()
_semantic_model = None
_schema_embeddings_cache = {}

def _build_caches():
    """Builds static lookup dictionaries exactly once."""
    if _dataset_terms_cache:
        return
        
    for d_type, schema in SCHEMAS.items():
        terms_map = {}
        for col in schema:
            terms_map[col.lower()] = 3.0 if col.lower() in HIGH_WEIGHT_TERMS else 1.0
            _all_valid_terms.add(col.lower())
            
        for canonical, alias_list in ALIASES[d_type].items():
            weight = 3.0 if canonical.lower() in HIGH_WEIGHT_TERMS else 1.0
            for alias in alias_list:
                terms_map[alias.lower()] = weight
                _all_valid_terms.add(alias.lower())
                
        _dataset_terms_cache[d_type] = terms_map

_build_caches()

def _get_semantic_model() -> SentenceTransformer:
    global _semantic_model
    if _semantic_model is None:
        logger.info("Loading SentenceTransformer model for semantic matching...")
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _semantic_model

def get_all_valid_terms() -> Set[str]:
    """Returns the globally cached set of valid canonical and alias terms."""
    return _all_valid_terms

def get_dataset_terms(dataset_type: str) -> Dict[str, float]:
    """Returns the cached mapping of terms to weights for a given dataset type."""
    return _dataset_terms_cache.get(dataset_type, {})

def detect_dataset_type(headers: List[str]) -> str:
    """
    Scores datasets using a combination of exact and fuzzy matching, applying 
    multipliers for highly unique identifying fields to prevent misclassification.
    """
    scores = {"bank": 0.0, "cdr": 0.0, "ipdr": 0.0}
    clean_headers = [str(h).lower().strip() for h in headers if pd.notna(h) and str(h).strip()]
    
    for dataset in SCHEMAS.keys():
        terms_dict = _dataset_terms_cache[dataset]
        valid_terms = list(terms_dict.keys())
        
        for header in clean_headers:
            match = process.extractOne(header, valid_terms, scorer=fuzz.WRatio)
            if match and match[1] >= 80:
                matched_term = match[0]
                confidence_score = match[1]
                weight = terms_dict[matched_term]
                scores[dataset] += (confidence_score * weight)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_dataset, best_score = sorted_scores[0]
    runner_up_dataset, runner_up_score = sorted_scores[1]

    if best_score < 160:
        raise UnknownDatasetError(f"Cannot identify dataset. Confidence too low. Scores: {scores}")
        
    if runner_up_score > 0 and (best_score - runner_up_score) <= 80:
        raise AmbiguousDatasetError(
            f"Dataset type ambiguous. Closely matched {best_dataset} ({best_score:.1f}) "
            f"and {runner_up_dataset} ({runner_up_score:.1f})."
        )

    logger.info(f"Detected dataset type '{best_dataset.upper()}' with score {best_score:.1f}")
    return best_dataset

def semantic_match(header: str, candidates: List[str], threshold: float = 0.55) -> Optional[str]:
    """Uses cached Sentence-Transformers embeddings to find the most semantically similar field."""
    if not candidates:
        return None
        
    model = _get_semantic_model()
    cache_key = tuple(candidates)
    
    if cache_key not in _schema_embeddings_cache:
        _schema_embeddings_cache[cache_key] = model.encode(candidates, convert_to_tensor=True)
        
    candidate_embs = _schema_embeddings_cache[cache_key]
    header_emb = model.encode(header, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(header_emb, candidate_embs)[0]
    best_idx = int(pd.Series(cos_scores.cpu().numpy()).idxmax())
    best_score = cos_scores[best_idx].item()
    
    if best_score >= threshold:
        return candidates[best_idx]
    return None

def find_best_match(header: str, dataset_type: str, used_canonical: Set[str]) -> Optional[str]:
    """Executes the optimized 4-tier matching pipeline for a single header."""
    header_clean = str(header).strip().lower()
    schema = SCHEMAS[dataset_type]
    aliases = ALIASES.get(dataset_type, {})
    
    available_schema = [c for c in schema if c not in used_canonical]
    if not available_schema:
        return None

    # Tier 1: Exact Match
    for canonical in available_schema:
        if canonical.lower() == header_clean:
            return canonical

    # Tier 2: Alias Match
    for canonical, alias_list in aliases.items():
        if canonical in available_schema:
            for alias in alias_list:
                if alias.lower() == header_clean:
                    return canonical

    # Tier 3: RapidFuzz Match (Syntactic Similarity)
    match_result = process.extractOne(header_clean, available_schema, scorer=fuzz.WRatio)
    if match_result:
        matched_str, score, _ = match_result
        if score >= 80.0:
            return matched_str

    # Tier 4: Semantic Match (Contextual Similarity)
    semantic_result = semantic_match(header_clean, available_schema, threshold=0.55)
    if semantic_result:
        return semantic_result

    return None

def map_columns(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Maps the extracted DataFrame columns to the canonical schema."""
    mapping = {}
    used_canonical = set()
    
    for col in df.columns:
        if pd.isna(col) or 'unnamed' in str(col).lower():
            continue
            
        best_match = find_best_match(col, dataset_type, used_canonical)
        if best_match:
            mapping[col] = best_match
            used_canonical.add(best_match)
        else:
            logger.warning(f"Unmapped column '{col}' could not be matched to any canonical field.")
            
    df_mapped = df.rename(columns=mapping)
    schema_cols = SCHEMAS[dataset_type]
    # Keep canonical cols AND Transaction_Amount_Merged if present (used by fallback)
    keep = [col for col in df_mapped.columns
            if col in schema_cols or col == "Transaction_Amount_Merged"]
    return df_mapped[keep]

def ensure_schema(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Ensures exact canonical schema layout. Fills missing with pd.NA."""
    canonical_schema = SCHEMAS[dataset_type]
    return df.reindex(columns=canonical_schema, fill_value=pd.NA)

def save_csv(df: pd.DataFrame, dataset_type: str, output_dir: str = ".") -> str:
    """Saves DataFrame complying with the strict CSV contract (UTF-8, No Index)."""
    import os
    filename = f"{dataset_type}_parsed.csv"
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Successfully saved structured CSV to: {output_path}")
    return output_path