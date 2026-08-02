"""
pdf_parser.py

Handles PDF ingestion, advanced table & text extraction fallback, robust multi-page 
header detection, and data normalization, acting as the primary entry point.
"""

import pdfplumber
import pandas as pd
import unicodedata
import os
import re
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict

from rapidfuzz import fuzz, process

from schema_mapper import (
    detect_dataset_type, 
    map_columns, 
    ensure_schema, 
    save_csv,
    get_all_valid_terms,
    get_dataset_terms,
    semantic_match,
    SCHEMAS
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScannedPDFError(Exception):
    pass

class PDFExtractionError(Exception):
    pass

# ==========================================
# KNOWN BANKS & NORMALIZATION
# ==========================================

_KNOWN_BANKS = [
    "AXIS BANK", "STATE BANK OF INDIA", "HDFC BANK", "ICICI BANK",
    "PUNJAB NATIONAL BANK", "BANK OF BARODA", "CANARA BANK",
    "UNION BANK OF INDIA", "INDIAN BANK", "IDFC FIRST BANK",
    "KOTAK MAHINDRA BANK", "YES BANK", "BANK OF INDIA",
    "CENTRAL BANK OF INDIA", "INDIAN OVERSEAS BANK", "UCO BANK",
    "BANK OF MAHARASHTRA", "CITY UNION BANK", "FEDERAL BANK",
    "SOUTH INDIAN BANK", "KARUR VYSYA BANK", "BANDHAN BANK",
    "RBL BANK", "INDUSIND BANK", "DBS BANK", "CITI BANK",
    "STANDARD CHARTERED BANK", "HSBC BANK", "JANA SMALL FINANCE BANK",
    "AU SMALL FINANCE BANK", "ESAF SMALL FINANCE BANK",
    "EQUITAS SMALL FINANCE BANK", "UJJIVAN SMALL FINANCE BANK",
    "NORTH EAST SMALL FINANCE BANK", "SHIVALIK SMALL FINANCE BANK",
    "UNITY SMALL FINANCE BANK", "FINCARE SMALL FINANCE BANK",
    "ANDHRA BANK", "CORPORATION BANK", "ORIENTAL BANK OF COMMERCE",
    "SYNDICATE BANK", "VIJAYA BANK", "DENA BANK", "ALLAHABAD BANK",
    "UNITED BANK OF INDIA"
]

_BANK_NAME_NORMALIZATION = {
    "AXIS BANK": "Axis Bank",
    "STATE BANK OF INDIA": "State Bank of India",
    "HDFC BANK": "HDFC Bank",
    "ICICI BANK": "ICICI Bank",
    "PUNJAB NATIONAL BANK": "Punjab National Bank",
    "BANK OF BARODA": "Bank of Baroda",
    "CANARA BANK": "Canara Bank",
    "UNION BANK OF INDIA": "Union Bank of India",
    "INDIAN BANK": "Indian Bank",
    "IDFC FIRST BANK": "IDFC First Bank",
    "KOTAK MAHINDRA BANK": "Kotak Mahindra Bank",
    "YES BANK": "Yes Bank",
    "BANK OF INDIA": "Bank of India",
    "CENTRAL BANK OF INDIA": "Central Bank of India",
    "INDIAN OVERSEAS BANK": "Indian Overseas Bank",
    "UCO BANK": "UCO Bank",
    "BANK OF MAHARASHTRA": "Bank of Maharashtra",
    "CITY UNION BANK": "City Union Bank",
    "FEDERAL BANK": "Federal Bank",
    "SOUTH INDIAN BANK": "South Indian Bank",
    "KARUR VYSYA BANK": "Karur Vysya Bank",
    "BANDHAN BANK": "Bandhan Bank",
    "RBL BANK": "RBL Bank",
    "INDUSIND BANK": "IndusInd Bank",
    "DBS BANK": "DBS Bank",
    "CITI BANK": "Citi Bank",
    "STANDARD CHARTERED BANK": "Standard Chartered Bank",
    "HSBC BANK": "HSBC Bank",
}

_PROVIDER_KEYWORDS = {
    "AXIS BANK": ["AXIS"],
    "STATE BANK OF INDIA": ["SBI"],
    "HDFC BANK": ["HDFC"],
    "ICICI BANK": ["ICICI"],
    "PUNJAB NATIONAL BANK": ["PNB"],
    "BANK OF BARODA": ["BOB"],
    "CANARA BANK": ["CANARA"],
    "UNION BANK OF INDIA": ["UNION"],
    "INDIAN BANK": ["INDIAN BANK"],
    "IDFC FIRST BANK": ["IDFC"],
    "KOTAK MAHINDRA BANK": ["KOTAK"],
    "YES BANK": ["YES BANK"],
    "BANK OF INDIA": ["BANK OF INDIA"],
}

def _normalize_text(text: str) -> str:
    """Removes strange unicode characters, newlines, tabs, and trims whitespace."""
    if pd.isna(text):
        return text
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==========================================
# METADATA EXTRACTION HELPERS
# ==========================================

def _find_bank_name(text: str, text_upper: str) -> Optional[str]:
    """
    Finds a known bank name in the text using word boundaries.
    Uses IFSC code as the authoritative source when present, because the
    transaction particulars often contain counterparty bank names (e.g. "Yes Bank"
    in a UPI transfer) that must NOT be reported as the sender's bank.
    Returns None if no known bank is found.
    """
    # --- Priority 1: Derive from IFSC prefix (most reliable) ---
    ifsc_match = re.search(r'(?:IFSC|IFSC\s*Code)\s*[:.]?\s*([A-Za-z]{4}0[A-Za-z0-9]{6})',
                           text, re.IGNORECASE)
    if ifsc_match:
        prefix = ifsc_match.group(1)[:4].upper()
        ifsc_to_bank = {
            "UTIB": "Axis Bank",       "SBIN": "State Bank of India",
            "HDFC": "HDFC Bank",       "ICIC": "ICICI Bank",
            "KKBK": "Kotak Mahindra Bank", "PUNB": "Punjab National Bank",
            "BARB": "Bank of Baroda",  "CNRB": "Canara Bank",
            "UBIN": "Union Bank of India", "IDFB": "IDFC First Bank",
            "YESB": "Yes Bank",        "BKID": "Bank of India",
            "INDB": "IndusInd Bank",   "RATN": "RBL Bank",
        }
        bank = ifsc_to_bank.get(prefix)
        if bank:
            return bank

    # --- Priority 2: Explicit bank name in header text only (first 800 chars) ---
    # Restrict search to the document header to avoid picking up counterparty
    # bank names embedded in transaction particulars rows.
    header_text = text_upper[:800]
    found_banks = {}
    for bank in _KNOWN_BANKS:
        pattern = r'\b' + re.escape(bank) + r'\b'
        matches = list(re.finditer(pattern, header_text))
        if matches:
            found_banks[bank] = len(matches)

    if found_banks:
        best_bank = max(found_banks, key=found_banks.get)
        return _BANK_NAME_NORMALIZATION.get(best_bank, best_bank.title())

    return None

def _detect_provider_from_text(text_upper: str) -> Optional[str]:
    """
    Fallback provider detection from raw text when no explicit bank name is found.
    """
    for bank_name, keywords in _PROVIDER_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_upper:
                return _BANK_NAME_NORMALIZATION.get(bank_name, bank_name.title())
    return None

def _find_account_number(text: str) -> Optional[str]:
    """Extracts account number from statement text using multiple strategies."""
    # Strategy 1: Label-based extraction with flexible spacing
    label_patterns = [
        r'(?i)(?:a/c|account|acct|sb|savings?|current)\s*(?:a/c)?\s*(?:no|number|#)?\s*[:.]?\s*(\d{10,20})\b',
        r'(?i)account\s*(?:number|no\.?|#)\s*[:.]?\s*(\d{10,20})\b',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) >= 6:
                return candidate
    
    # Strategy 2: Contextual patterns for masked/unmasked numbers
    context_patterns = [
        r'(?i)account\s*(?:number|no)?\s*[:.]?\s*([X\d]{4,20})\b',
        r'(?i)a/c\s*(?:no)?\s*[:.]?\s*([X\d]{4,20})\b',
    ]
    
    for pattern in context_patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) >= 6 and any(c.isdigit() for c in candidate):
                return candidate
    
    return None

def _find_ifsc(text: str) -> Optional[str]:
    """Extracts IFSC code from text using label and standalone patterns."""
    # Strategy 1: Explicit IFSC label
    label_pattern = r'(?:IFSC|IFSC\s*CODE|IFSC:)\s*[:.]?\s*([A-Za-z]{4}0[A-Za-z0-9]{6})'
    match = re.search(label_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Strategy 2: Standalone IFSC pattern with known prefix validation
    standalone_pattern = r'\b([A-Za-z]{4}0[A-Za-z0-9]{6})\b'
    matches = re.findall(standalone_pattern, text)
    
    known_prefixes = {
        'UTIB', 'SBIN', 'HDFC', 'ICIC', 'PUNB', 'BARB', 'CNRB', 'UBIN', 
        'IDFB', 'KKBK', 'YESB', 'BKID', 'IOBA', 'MAHB', 'SYNB', 'ALLA', 
        'ANDH', 'CBIN', 'CORP', 'DEN', 'FED', 'INDB', 'JAKA', 'KARB', 
        'KNSB', 'LAVB', 'MEGH', 'NKGS', 'OIBA', 'PSIB', 'RATN', 'SIBL', 
        'TMBL', 'TNSC', 'UBSW', 'UCBA', 'UTBI', 'VIJB', 'VYSA'
    }
    
    for m in matches:
        if m[:4].upper() in known_prefixes:
            return m.upper()
    
    # Fallback: return first match if any
    if matches:
        return matches[0].upper()
    
    return None

def _find_customer_name(text: str) -> Optional[str]:
    """Extracts customer/account holder name from statement text."""
    patterns = [
        r'(?:Customer\s*Name|Account\s*Holder|Holder\s*Name|Customer)\s*[:.]?\s*(Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Za-z][A-Za-z\s\.]{1,40})',
        r'(?:Name\s*[:.]\s*)(Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s*([A-Za-z][A-Za-z\s\.]{1,40})',
        r'(?:Customer\s*Name|Account\s*Holder)\s*[:.]?\s*([A-Za-z][A-Za-z\s\.]{2,40})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if match.lastindex == 2:
                name = (match.group(1) or "") + " " + match.group(2)
            else:
                name = match.group(1)
            
            name = re.sub(r'\s+', ' ', name).strip()
            # Remove trailing non-name words and state codes
            name = re.sub(r'\s+(Branch|IFSC|Account|Date|Period|Address|City|State|No|Number|Code|ID|Type)\s*$', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+[\[\(][A-Z]{2}[\]\)]\s*$', '', name)
            name = name.strip()
            
            if len(name) >= 3 and not re.match(r'^\d+$', name):
                if not any(word in name.upper() for word in ['BRANCH', 'IFSC', 'ACCOUNT', 'STATEMENT', 'BALANCE', 'BANK']):
                    return name.title()
    return None

def _find_account_type(text_upper: str) -> Optional[str]:
    """Extracts account type from text."""
    type_map = {
        'Savings': [r'\bSAVINGS\b', r'\bSAVINGS\s+ACCOUNT\b', r'\bSB\s+A\/C\b', r'\bSAV\s+A\/C\b'],
        'Current': [r'\bCURRENT\b', r'\bCURRENT\s+ACCOUNT\b', r'\bCA\s+A\/C\b'],
        'Salary': [r'\bSALARY\b', r'\bSALARY\s+ACCOUNT\b'],
        'OD': [r'\bOD\b', r'\bOVERDRAFT\b', r'\bO\.D\.'],
        'CC': [r'\bCC\b', r'\bCASH\s+CREDIT\b', r'\bC\.C\.'],
        'NRE': [r'\bNRE\b', r'\bNRE\s+ACCOUNT\b'],
        'NRO': [r'\bNRO\b', r'\bNRO\s+ACCOUNT\b'],
    }
    
    for acc_type, patterns in type_map.items():
        for pattern in patterns:
            if re.search(pattern, text_upper):
                return acc_type
    return None

def _find_customer_id(text: str) -> Optional[str]:
    """Extracts customer ID from text."""
    patterns = [
        r'(?:Customer\s*ID|Cust\s*ID|CIF|CRN|Customer\s*Number)\s*[:.]?\s*(\d{6,16})',
        r'(?:CIF\s*No|Cust\s*ID\s*No|CRN\s*No)\s*[:.]?\s*(\d{6,16})',
        r'(?:Customer\s*ID|Cust\s*ID|CIF|CRN)\s*[:.]?\s*(\d{6,16})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def _find_branch(text: str) -> Optional[str]:
    """Extracts branch name from text. Never used as bank name."""
    patterns = [
        r'(?:Branch|Branch\s*Name)\s*[:.]?\s*([A-Za-z][A-Za-z0-9\s,\-\[\]\(\)]{2,50})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            branch = match.group(1).strip()
            branch = re.sub(r'\s*(IFSC|Account|Date|Period|Address|City|State|Code|No|Number|Pin)\s*.*$', '', branch, flags=re.IGNORECASE)
            branch = branch.strip()
            if len(branch) >= 3:
                return branch
    return None

def _extract_statement_metadata(pdf_path: str) -> Dict[str, str]:
    """
    Extracts statement metadata from the first page of a bank statement PDF.
    Searches both free text and table cells comprehensively.
    Returns a dictionary keyed by canonical column names.
    """
    metadata: Dict[str, str] = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return metadata
                
            page = pdf.pages[0]
            
            # Extract free text
            free_text = page.extract_text() or ""
            
            # Extract table text
            table_text = ""
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    if row:
                        table_text += " ".join(str(cell) for cell in row if cell is not None) + " "
            
            # Combine both sources
            all_text = free_text + "\n" + table_text
            all_text_upper = all_text.upper()
            
            # Bank Name (highest priority — never allow branch names)
            bank_name = _find_bank_name(all_text, all_text_upper)
            if not bank_name:
                bank_name = _detect_provider_from_text(all_text_upper)
            if bank_name:
                metadata["Sender_Bank_Name"] = bank_name
                logger.info(f"Metadata: Bank Name = {bank_name}")
            
            # Account Number
            acc_num = _find_account_number(all_text)
            if acc_num:
                metadata["Sender_Account_Number"] = acc_num
                logger.info(f"Metadata: Account Number = {acc_num}")
            
            # IFSC
            ifsc = _find_ifsc(all_text)
            if ifsc:
                metadata["Sender_IFSC"] = ifsc
                logger.info(f"Metadata: IFSC = {ifsc}")
            
            # Customer Name
            cust_name = _find_customer_name(all_text)
            if cust_name:
                metadata["Sender_Customer_Name"] = cust_name
                logger.info(f"Metadata: Customer Name = {cust_name}")
            
            # Account Type
            acc_type = _find_account_type(all_text_upper)
            if acc_type:
                metadata["Sender_Account_Type"] = acc_type
                logger.info(f"Metadata: Account Type = {acc_type}")
            
            # Customer ID
            cust_id = _find_customer_id(all_text)
            if cust_id:
                metadata["Sender_Customer_ID"] = cust_id
                logger.info(f"Metadata: Customer ID = {cust_id}")
            
            # Currency (default INR for Indian bank statements)
            metadata["Currency"] = "INR"
            
            # Branch (extracted for logging only, never injected as bank name)
            branch = _find_branch(all_text)
            if branch:
                logger.info(f"Metadata: Branch = {branch}")
                
    except Exception as e:
        logger.warning(f"Statement metadata extraction failed: {e}")
        
    return metadata

def _apply_statement_metadata(df: pd.DataFrame, metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Injects statement metadata into every transaction row.
    Overwrites any incorrectly extracted values (e.g. branch names 
    mapped to Sender_Bank_Name) with the correct statement-level metadata.
    """
    if not metadata:
        return df
        
    for col, value in metadata.items():
        if value and pd.notna(value):
            df[col] = value
            
    return df

def _detect_provider_metadata(df: pd.DataFrame) -> Optional[str]:
    """
    Detects potential data providers from the DataFrame content.
    Returns the primary bank/provider name for metadata fallback.
    """
    provider_map = {
        "SBI": "State Bank of India",
        "HDFC": "HDFC Bank",
        "ICICI": "ICICI Bank",
        "AXIS": "Axis Bank",
        "PNB": "Punjab National Bank",
        "BOB": "Bank of Baroda",
        "CANARA": "Canara Bank",
        "UNION": "Union Bank of India",
        "IDFC": "IDFC First Bank",
        "KOTAK": "Kotak Mahindra Bank",
        "YES BANK": "Yes Bank",
        "INDIAN BANK": "Indian Bank",
        "AIRTEL": "Airtel",
        "JIO": "Jio",
        "VI": "Vodafone Idea",
        "BSNL": "BSNL"
    }
    
    text_dump = " ".join(df.head(20).fillna("").astype(str).values.flatten()).upper()
    found_scores = {}
    
    for keyword, full_name in provider_map.items():
        if keyword.upper() in text_dump:
            found_scores[full_name] = found_scores.get(full_name, 0) + 1
    
    if found_scores:
        best_provider = max(found_scores, key=found_scores.get)
        logger.info(f"Detected Provider Context: {best_provider}")
        return best_provider
        
    return None

def _extract_via_text_fallback(pdf) -> pd.DataFrame:
    """Fallback mechanism: extracts text, splits by lines, then by multiple spaces."""
    all_rows = []
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Split by 2 or more spaces
            row = re.split(r'\s{2,}', line)
            if len(row) > 1:
                all_rows.append(row)
                
    if not all_rows:
        return pd.DataFrame()
        
    # Pad rows to consistent length
    max_cols = max(len(r) for r in all_rows)
    padded = [r + [pd.NA] * (max_cols - len(r)) for r in all_rows]
    return pd.DataFrame(padded)

def extract_tables_from_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Extracts tables across all pages using pdfplumber strategies.
    Falls back to regex-based text extraction if tabular bounds fail.
    Raises ScannedPDFError if the PDF appears to lack textual content.
    """
    all_rows = []
    total_chars = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        # Pre-check for scanned PDFs
        for page in pdf.pages:
            t = page.extract_text()
            if t: total_chars += len(t)
        
        if total_chars < 50:
            raise ScannedPDFError("PDF contains almost no text. It is likely a scanned image requiring OCR.")
            
        # Strategy 1: Default tables
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        all_rows.append(row)
        
        # Strategy 2: Explicit Text strategy
        if not all_rows:
            logger.info("Default table extraction failed. Attempting alternative text-based tabular strategy...")
            table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
            for page in pdf.pages:
                tables = page.extract_tables(table_settings)
                if tables:
                    for table in tables:
                        for row in table:
                            all_rows.append(row)

        # Strategy 3: Pure text extraction splitting
        if not all_rows:
            logger.info("Structured table parsing failed. Attempting spatial text extraction fallback...")
            df = _extract_via_text_fallback(pdf)
            if not df.empty:
                return df
                
    if not all_rows:
        raise PDFExtractionError("No tabular data could be extracted from the provided PDF.")

    return pd.DataFrame(all_rows)

def _detect_and_apply_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans candidate rows using fuzzy similarity against all known schema concepts.
    Cleans bracketed text and uses semantic matching as a final fallback.
    """
    all_valid_terms = get_all_valid_terms()
    best_idx = -1
    max_score = 0
    limit = min(25, len(df))

    for i in range(limit):
        row = df.iloc[i].dropna().astype(str).str.lower().str.strip()
        score = 0
        for cell in row:
            if not cell:
                continue
            # Remove brackets and colon punctuation for cleaner matching
            clean_cell = re.sub(r'[\[\]\(\)\{\}\:\;]', '', cell).strip()
            
            match = process.extractOne(clean_cell, all_valid_terms, scorer=fuzz.WRatio)
            if match and match[1] >= 80:
                score += match[1]
            elif len(clean_cell) > 3:
                # Semantic fallback for the cell
                sem_match = semantic_match(clean_cell, list(all_valid_terms), threshold=0.65)
                if sem_match:
                    score += 65

        if score > max_score:
            max_score = score
            best_idx = i

    if max_score < 160: 
        raise ValueError(f"Failed to detect a valid table header row. Max score achieved: {max_score}")

    logger.info(f"Header row detected at index {best_idx} (Score: {max_score:.1f})")
    df.columns = df.iloc[best_idx].astype(str).apply(_normalize_text)
    return df.iloc[best_idx+1:].reset_index(drop=True)

def _remove_repeated_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes header rows that repeat on subsequent pages by comparing individual cells 
    against the detected header to avoid monolithic string concatenation flaws.
    """
    header_cells = [str(c).strip().lower() for c in df.columns]
    
    def is_repeated_header(row):
        matches = 0
        valid_cells = 0
        for h_str, r_cell in zip(header_cells, row):
            r_str = str(r_cell).strip().lower()
            if h_str and h_str != 'nan':
                valid_cells += 1
                # Ignore minor punctuation/spacing
                h_clean = re.sub(r'\W+', '', h_str)
                r_clean = re.sub(r'\W+', '', r_str)
                if r_clean and fuzz.WRatio(h_clean, r_clean) > 85:
                    matches += 1
                    
        if valid_cells == 0:
            return False
        return (matches / valid_cells) >= 0.7 

    mask = df.apply(is_repeated_header, axis=1)
    return df[~mask].reset_index(drop=True)

def _merge_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Merges isolated Debit/Credit columns or Amount+DR/CR into signed Transaction_Amount."""
    cols_lower = [str(c).lower().strip() for c in df.columns]
    
    # Strategy 1: Separate debit and credit columns
    debit_col, credit_col = None, None
    debit_aliases = ['debit', 'withdrawal', 'dr']
    credit_aliases = ['credit', 'deposit', 'cr']

    for original_col, lower_col in zip(df.columns, cols_lower):
        if lower_col in debit_aliases or any(fuzz.WRatio(a, lower_col) > 90 for a in debit_aliases):
            debit_col = original_col
        elif lower_col in credit_aliases or any(fuzz.WRatio(a, lower_col) > 90 for a in credit_aliases):
            credit_col = original_col

    if debit_col and credit_col:
        s_debit = pd.to_numeric(df[debit_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        s_credit = pd.to_numeric(df[credit_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df['Transaction_Amount_Merged'] = s_credit.fillna(0) - s_debit.fillna(0)
        # Remove zero amounts (both empty)
        df.loc[(s_debit.isna()) & (s_credit.isna()), 'Transaction_Amount_Merged'] = pd.NA
        logger.info("Merged isolated Debit and Credit columns into signed amount.")
        return df

    # Strategy 2: Single Amount column + DR/CR flag column
    amount_col = None
    drcr_col = None
    for original_col, lower_col in zip(df.columns, cols_lower):
        lower_clean = re.sub(r'[\s\(\)]', '', lower_col)
        if "amount" in lower_clean and ("inr" in lower_clean or "rs" in lower_clean):
            amount_col = original_col
        if lower_col in {"dr/cr", "dr_cr", "drcr", "type", "dr / cr", "dr /cr"}:
            drcr_col = original_col

    if amount_col and drcr_col:
        def to_signed(row):
            try:
                val = float(str(row[amount_col]).replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "").strip())
            except (ValueError, TypeError):
                return pd.NA
            flag = str(row[drcr_col]).strip().upper()
            if flag in {"DR", "D", "DEBIT", "DR."}:
                return -abs(val)
            if flag in {"CR", "C", "CREDIT", "CR."}:
                return abs(val)
            return val

        df['Transaction_Amount_Merged'] = df.apply(to_signed, axis=1)
        logger.info("Merged Amount(INR) and DR/CR into signed Transaction_Amount_Merged.")
    
    # Strategy 3: Single Amount column with DR/CR embedded in the same cell
    elif amount_col:
        def extract_signed(val):
            if pd.isna(val):
                return pd.NA
            s = str(val).strip()
            # Check for DR/CR suffix
            dr_match = re.search(r'([\d,]+\.?\d*)\s*(DR|DEBIT)', s, re.IGNORECASE)
            cr_match = re.search(r'([\d,]+\.?\d*)\s*(CR|CREDIT)', s, re.IGNORECASE)
            if dr_match:
                return -abs(float(dr_match.group(1).replace(",", "")))
            if cr_match:
                return abs(float(cr_match.group(1).replace(",", "")))
            # Try plain number
            try:
                return float(re.sub(r'[^\d.]', '', s))
            except ValueError:
                return pd.NA
        
        df['Transaction_Amount_Merged'] = df[amount_col].apply(extract_signed)
        logger.info("Extracted signed amount from single Amount column.")

    return df


def _parse_transaction_particulars(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Transaction Particulars to extract mode, ID, beneficiary, and bank."""
    if "Transaction_Mode" not in df.columns:
        return df
    
    def parse_particulars(text):
        if pd.isna(text):
            return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
        
        s = str(text).strip()
        parts = [p.strip() for p in s.split('/') if p.strip()]
        
        mode = None
        txn_id = pd.NA
        beneficiary = pd.NA
        bank = pd.NA
        
        if not parts:
            return pd.Series([s, pd.NA, pd.NA, pd.NA])
        
        # Detect mode from first part
        mode_keywords = {
            'IMPS': 'IMPS', 'NEFT': 'NEFT', 'RTGS': 'RTGS', 'UPI': 'UPI',
            'POS': 'POS', 'EDC': 'EDC', 'ATM': 'ATM', 'CASH': 'CASH',
            'CHQ': 'CHEQUE', 'CHEQUE': 'CHEQUE', 'DR': 'DEBIT_CARD',
            'CASHBACK': 'CASHBACK', 'INITIAL': 'INITIAL_FUNDING',
            'OPENING': 'OPENING_BALANCE', 'CLOSING': 'CLOSING_BALANCE'
        }
        
        first = parts[0].upper()
        for keyword, canonical in mode_keywords.items():
            if keyword in first:
                mode = canonical
                break
        
        if mode == 'IMPS' and len(parts) >= 2:
            mode = f"IMPS_{parts[1]}" if parts[1] in ('P2A', 'P2P', 'P2M') else 'IMPS'
        
        if mode in ('POS', 'EDC') and len(parts) >= 2:
            txn_id = parts[-1] if parts[-1].isdigit() else pd.NA
        
        if mode == 'IMPS' and len(parts) >= 3:
            txn_id = parts[2] if parts[2].isdigit() else pd.NA
            # Look for bank name in parts
            for part in parts[3:]:
                if any(b in part.upper() for b in ['BANK', 'IDFC', 'HDFC', 'SBI', 'ICICI', 'AXIS', 'PNB']):
                    bank = part
                    break
            # Beneficiary is usually the part after ID
            if len(parts) >= 4 and not any(b in parts[3].upper() for b in ['BANK', 'X0']):
                beneficiary = parts[3]
        
        # If no mode detected, keep original as mode
        if mode is None:
            mode = s
        
        return pd.Series([mode, txn_id, beneficiary, bank])
    
    parsed = df["Transaction_Mode"].apply(parse_particulars)
    parsed.columns = ["Transaction_Mode_Clean", "Transaction_ID_Parsed", "Receiver_Customer_Name_Parsed", "Receiver_Bank_Name_Parsed"]
    
    # Only overwrite if we got meaningful parses
    df["Transaction_Mode"] = parsed["Transaction_Mode_Clean"]
    
    # Safely set Transaction_ID — create if missing
    if "Transaction_ID" not in df.columns:
        df["Transaction_ID"] = parsed["Transaction_ID_Parsed"]
    else:
        df["Transaction_ID"] = df["Transaction_ID"].fillna(parsed["Transaction_ID_Parsed"])
    
    # Safely set Receiver_Customer_Name — create if missing
    if "Receiver_Customer_Name" not in df.columns:
        df["Receiver_Customer_Name"] = parsed["Receiver_Customer_Name_Parsed"]
    else:
        df["Receiver_Customer_Name"] = df["Receiver_Customer_Name"].fillna(parsed["Receiver_Customer_Name_Parsed"])
    
    # Safely set Receiver_Bank_Name — create if missing
    if "Receiver_Bank_Name" not in df.columns:
        df["Receiver_Bank_Name"] = parsed["Receiver_Bank_Name_Parsed"]
    else:
        df["Receiver_Bank_Name"] = df["Receiver_Bank_Name"].fillna(parsed["Receiver_Bank_Name_Parsed"])
    
    return df


def _remove_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove opening balance, closing balance, and total rows."""
    if "Transaction_Mode" not in df.columns:
        return df
    
    summary_patterns = [
        r'^\s*OPENING\s+BALANCE\s*$',
        r'^\s*CLOSING\s+BALANCE\s*$',
        r'^\s*TRANSACTION\s+TOTAL\s*',
        r'^\s*BROUGHT\s+FORWARD\s*$',
        r'^\s*CARRIED\s+FORWARD\s*$',
    ]
    
    mask = df["Transaction_Mode"].astype(str).str.strip().str.upper().apply(
        lambda x: not any(re.match(p, x, re.IGNORECASE) for p in summary_patterns)
    )
    
    removed = len(df) - mask.sum()
    if removed > 0:
        logger.info(f"Removed {removed} summary rows (opening/closing balance, totals).")
    
    return df[mask].reset_index(drop=True)


def _clean_raw_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Applies robust, pre-mapping structural normalizations safely."""
    provider = _detect_provider_metadata(df)
    df = _detect_and_apply_header(df)
    df = _remove_repeated_headers(df)
    
    # Normalize string format correctly depending on pandas version
    try:
        df = df.map(lambda x: _normalize_text(x) if isinstance(x, str) else x)
    except AttributeError:
        df = df.applymap(lambda x: _normalize_text(x) if isinstance(x, str) else x)
        
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    df = _merge_debit_credit(df)
    
    return df.reset_index(drop=True), provider

def _clean_mapped_dataframe(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Cleans mapped numeric and date values securely.
    STRICTLY avoids modifying Identifiers.
    """
    numeric_columns = {
        "bank": ["Transaction_Amount", "Transaction_Amount_Merged"],
        "cdr": ["Call_Duration_Seconds"],
        "ipdr": ["Session_Duration_Seconds"]
    }
    date_columns = {
        "bank": ["Date"],
        "cdr": ["Call_Date"],
        "ipdr": ["Session_Date"]
    }
    
    # Clean monetary/duration fields robustly
    for col in numeric_columns.get(dataset_type, []):
        if col in df.columns:
            def _clean_num(val):
                if pd.isna(val): 
                    return val
                s = str(val).replace(',', '').replace('₹', '').replace('$', '').replace('Rs.', '').replace('Rs', '')
                s = re.sub(r'(?i)\bcr\b|\bdr\b', '', s).strip()
                return s if s else pd.NA
                
            df[col] = df[col].apply(_clean_num)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalize canonical date formats consistently
    for col in date_columns.get(dataset_type, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True).dt.strftime('%Y-%m-%d')
            df[col] = df[col].replace({np.nan: pd.NA, 'NaT': pd.NA})

    # Bank-specific post-processing
    if dataset_type == "bank":
        # Fallback: copy merged amount to Transaction_Amount if latter is empty/missing
        if "Transaction_Amount_Merged" in df.columns:
            if "Transaction_Amount" not in df.columns:
                df["Transaction_Amount"] = df["Transaction_Amount_Merged"]
            else:
                df["Transaction_Amount"] = df["Transaction_Amount"].fillna(df["Transaction_Amount_Merged"])
        
        df = _parse_transaction_particulars(df)
        df = _remove_summary_rows(df)

    return df

def _validate_schema(df: pd.DataFrame, dataset_type: str):
    """Raises robust validation errors ensuring output structural integrity."""
    critical_fields = {
        "bank": ["Date", "Transaction_Amount"],
        "cdr": ["Call_Date", "A_Party_Number"],
        "ipdr": ["Session_Date", "Source_IP_Address"]
    }
    
    if df.empty or df.isna().all(axis=None):
        raise ValueError("Validation Failed: Extracted DataFrame is entirely empty or null.")

    for col in critical_fields.get(dataset_type, []):
        if col not in df.columns:
            # Fallback for merged amount
            if col == "Transaction_Amount" and "Transaction_Amount_Merged" in df.columns:
                df["Transaction_Amount"] = df["Transaction_Amount_Merged"]
                continue
            raise ValueError(f"Validation Failed: Critical column '{col}' is entirely missing from schema.")
        if df[col].isna().all():
            # Fallback for merged amount
            if col == "Transaction_Amount" and "Transaction_Amount_Merged" in df.columns:
                df["Transaction_Amount"] = df["Transaction_Amount_Merged"]
                if not df[col].isna().all():
                    continue
            raise ValueError(f"Validation Failed: Critical column '{col}' is present but completely empty.")

def _print_parsing_summary(original_rows: int, retained_rows: int, 
                           dataset_type: str, provider: Optional[str], 
                           mapped_df: pd.DataFrame, final_csv_path: str):
    """Logs an operational summary of the parsing execution."""
    canonical = SCHEMAS[dataset_type]
    present = [c for c in canonical if c in mapped_df.columns and not mapped_df[c].isna().all()]
    missing = [c for c in canonical if c not in present]
    
    logger.info("=== PARSING SUMMARY ===")
    logger.info(f"Dataset Type     : {dataset_type.upper()}")
    logger.info(f"Provider Context : {provider if provider else 'Unknown'}")
    logger.info(f"Rows Extracted   : {original_rows}")
    logger.info(f"Rows Retained    : {retained_rows}")
    logger.info(f"Columns Mapped   : {len(present)}/{len(canonical)}")
    if missing:
        logger.info(f"Missing Columns  : {', '.join(missing)}")
    logger.info(f"Output Pathway   : {final_csv_path}")
    logger.info("=======================")

def parse_pdf(pdf_path: str, output_dir: str = ".") -> pd.DataFrame:
    """
    Main PDF Parsing Pipeline. Converts unstructured PDFs directly 
    into rigorously validated canonical CSVs suitable for downstream logic.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The PDF file at '{pdf_path}' was not found.")
        
    try:
        logger.info(f"Initiating parsing for: {os.path.basename(pdf_path)}")
        
        # 0. Statement Metadata Extraction
        statement_metadata = _extract_statement_metadata(pdf_path)
        
        # 1. Block Extraction
        raw_df = extract_tables_from_pdf(pdf_path)
        original_rows = len(raw_df)
        
        # 2. Structural & Header Normalization
        cleaned_df, provider = _clean_raw_dataframe(raw_df)
        retained_rows = len(cleaned_df)
        
        # 3. Intelligent Classification
        dataset_type = detect_dataset_type(list(cleaned_df.columns))
        
        # 4. Canonical Projection
        mapped_df = map_columns(cleaned_df, dataset_type)
        
        # 4.5 Metadata Injection for Bank Statements
        if dataset_type == "bank":
            # Fallback: use provider detection if explicit bank name not found
            if provider and "Sender_Bank_Name" not in statement_metadata:
                statement_metadata["Sender_Bank_Name"] = provider
            if statement_metadata:
                mapped_df = _apply_statement_metadata(mapped_df, statement_metadata)
        
        # 5. Semantic Value Normalization
        clean_mapped_df = _clean_mapped_dataframe(mapped_df, dataset_type)
        
        # 6. Strict Schema Enforcement
        final_df = ensure_schema(clean_mapped_df, dataset_type)
        
        # 7. Quality Sanity Validation
        _validate_schema(final_df, dataset_type)
        
        # 8. Dispatch & Summarize
        out_path = save_csv(final_df, dataset_type, output_dir)
        _print_parsing_summary(original_rows, retained_rows, dataset_type, provider, mapped_df, out_path)
        
        return final_df

    except Exception as e:
        logger.error(f"Failed to parse PDF '{pdf_path}': {str(e)}")
        raise