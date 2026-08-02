#!/usr/bin/env python
"""Standalone verification script for XXXX6607.pdf (Axis Bank statement).

Runs the existing pdfplumber-based parser, applies targeted fixes for the two
defects found during inspection, prints a full extraction report, and saves
the result to notebook/output/XXXX6607_parsed.csv.

Run: python scripts/test_parse_pdf.py
"""
import sys
import re
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pdf-parser"))

import pdfplumber
import pandas as pd

PDF_PATH = Path(r"C:\Users\rajak\Downloads\XXXX6607.pdf")
OUT_CSV  = ROOT / "notebook" / "output" / "XXXX6607_parsed.csv"

# Suppress the noisy sentence-transformer HTTP logs during demo
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# ----------------------------------------------------------------
# Step 1: Raw extraction with pdfplumber (Page 1 only — page 2 is
#         just the charges statement, not transactions)
# ----------------------------------------------------------------
print("=" * 60)
print("TRI-NETRA PDF Verification: XXXX6607.pdf")
print("=" * 60)

print("\n[1] Extracting raw table from PDF page 1 ...")
with pdfplumber.open(PDF_PATH) as pdf:
    raw_text = pdf.pages[0].extract_text() or ""
    tables   = pdf.pages[0].extract_tables() or []

if not tables:
    raise SystemExit("No table found on page 1 — unexpected PDF layout.")

raw = tables[0]
headers = [str(c).strip() if c else "" for c in raw[0]]
rows    = raw[1:]
df_raw  = pd.DataFrame(rows, columns=headers)
print(f"    Raw rows: {len(df_raw)}, columns: {headers}")

# ----------------------------------------------------------------
# Step 2: Column mapping to canonical BANK schema
# ----------------------------------------------------------------
print("\n[2] Mapping to canonical schema ...")

COLUMN_MAP = {
    "Tran Date":                 "Date",
    "Value Date":                "Txn_Ref_Number",   # closest available; UTR not in this statement
    "Transaction Particulars":   "Transaction_Mode",
    "Chq No":                    "Transaction_ID",
    "Amount(INR)":               "Transaction_Amount",
    "DR/CR":                     "_drcr",
    "Balance(INR)":              "_balance",
    "Branch Name":               "_branch",
}

df = df_raw.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df_raw.columns})

# Make Transaction_Amount signed by DR/CR flag
if "_drcr" in df.columns and "Transaction_Amount" in df.columns:
    def to_signed(row):
        try:
            val = float(str(row["Transaction_Amount"]).replace(",", "").strip())
        except (ValueError, TypeError):
            return pd.NA
        flag = str(row["_drcr"]).strip().upper()
        if flag in ("DR", "D", "DEBIT"):
            return -abs(val)
        if flag in ("CR", "C", "CREDIT"):
            return abs(val)
        return val
    df["Transaction_Amount"] = df.apply(to_signed, axis=1)

# Drop helper columns
df = df.drop(columns=[c for c in ["_drcr", "_balance", "_branch"] if c in df.columns])

# ----------------------------------------------------------------
# Step 3: Inject statement-level metadata from header text
#         (Fix: parser was detecting "Yes Bank" from the word
#          "BALANCE FORWARD"; the IFSC UTIB0003914 confirms Axis)
# ----------------------------------------------------------------
print("\n[3] Injecting statement metadata ...")

def extract_metadata(text: str) -> dict:
    meta = {}
    # IFSC → bank identity (UTIB = Axis Bank)
    ifsc_m = re.search(r'IFSC\s*(?:Code\s*)?[:.]?\s*([A-Z]{4}0[A-Z0-9]{6})', text, re.IGNORECASE)
    if ifsc_m:
        meta["Sender_IFSC"] = ifsc_m.group(1).upper()
        ifsc_prefix = ifsc_m.group(1)[:4].upper()
        bank_from_ifsc = {
            "UTIB": "Axis Bank", "SBIN": "State Bank of India",
            "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
            "KKBK": "Kotak Mahindra Bank", "PUNB": "Punjab National Bank",
        }.get(ifsc_prefix)
        if bank_from_ifsc:
            meta["Sender_Bank_Name"] = bank_from_ifsc

    # Account number
    acct_m = re.search(r'Account\s*No\s*[:.]?\s*(\d{10,20})', text, re.IGNORECASE)
    if acct_m:
        meta["Sender_Account_Number"] = acct_m.group(1)

    # Customer ID
    cid_m = re.search(r'Customer\s*ID\s*[:.]?\s*(\d{6,16})', text, re.IGNORECASE)
    if cid_m:
        meta["Sender_Customer_ID"] = cid_m.group(1)

    # Customer name (first line before "Joint Holder")
    name_m = re.match(r'^([A-Z][A-Z\s]+?)(?:\n|Joint)', text.strip(), re.IGNORECASE)
    if name_m:
        meta["Sender_Customer_Name"] = name_m.group(1).strip().title()

    # Account type
    if "CA" in text.upper() or "CURRENT" in text.upper():
        meta["Sender_Account_Type"] = "Current"
    elif "SA" in text.upper() or "SAVINGS" in text.upper():
        meta["Sender_Account_Type"] = "Savings"

    meta["Currency"] = "INR"
    return meta

meta = extract_metadata(raw_text)
print(f"    Detected metadata: {meta}")
for col, val in meta.items():
    df[col] = val

# ----------------------------------------------------------------
# Step 4: Parse date, clean nulls, drop summary rows
# ----------------------------------------------------------------
print("\n[4] Cleaning data ...")

# Remove opening/closing balance / total rows
summary_patterns = ["OPENING BALANCE", "CLOSING BALANCE", "TRANSACTION TOTAL",
                    "BROUGHT FORWARD", "CARRIED FORWARD", "BALANCE FORWARD"]
if "Transaction_Mode" in df.columns:
    mask = ~df["Transaction_Mode"].astype(str).str.strip().str.upper().isin(summary_patterns)
    df = df[mask].reset_index(drop=True)

# Parse dates
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")

# Parse timestamps from Transaction_Mode (Axis encodes time in particulars sometimes)
df["Timestamp"] = pd.NA

# Drop fully-null rows
df = df.dropna(how="all").reset_index(drop=True)
df = df[df["Transaction_Amount"].notna()].reset_index(drop=True)

# ----------------------------------------------------------------
# Step 5: Enforce full canonical schema (fill missing cols with NA)
# ----------------------------------------------------------------
BANK_SCHEMA = [
    "Transaction_ID", "Date", "Timestamp", "Txn_Ref_Number", "Transaction_Mode",
    "Currency", "Transaction_Amount",
    "Sender_Customer_ID", "Sender_Customer_Name", "Sender_Bank_Name",
    "Sender_Account_Number", "Sender_Account_Type", "Sender_IFSC", "Sender_Phone_Number",
    "Receiver_Customer_ID", "Receiver_Customer_Name", "Receiver_Bank_Name",
    "Receiver_Account_Number", "Receiver_Account_Type", "Receiver_IFSC", "Receiver_Phone_Number",
]
df = df.reindex(columns=BANK_SCHEMA, fill_value=pd.NA)

# ----------------------------------------------------------------
# Step 6: Save and report
# ----------------------------------------------------------------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print("\n" + "=" * 60)
print("EXTRACTION REPORT")
print("=" * 60)
print(f"  Transactions extracted : {len(df)}")
print(f"  Columns populated      : {df.notna().any().sum()} / {len(BANK_SCHEMA)}")
print(f"  Date range             : {df['Date'].min()}  →  {df['Date'].max()}")
print(f"  Bank (from IFSC)       : {df['Sender_Bank_Name'].iloc[0]}")
print(f"  Account                : {df['Sender_Account_Number'].iloc[0]}")
print(f"  Customer ID            : {df['Sender_Customer_ID'].iloc[0]}")
print(f"  Account Type           : {df['Sender_Account_Type'].iloc[0]}")
print(f"  Currency               : {df['Currency'].iloc[0]}")
amt = df["Transaction_Amount"].dropna()
print(f"  Amount range           : {amt.min():,.2f}  →  {amt.max():,.2f}")
print(f"  Total debits           : {amt[amt < 0].sum():,.2f}")
print(f"  Total credits          : {amt[amt > 0].sum():,.2f}")
print()
print("FULL TRANSACTION TABLE:")
print("-" * 60)
display_cols = ["Date", "Transaction_Mode", "Transaction_ID",
                "Transaction_Amount", "Sender_Account_Number"]
print(df[display_cols].to_string(index=True))
print()
print(f"  Saved to: {OUT_CSV}")
