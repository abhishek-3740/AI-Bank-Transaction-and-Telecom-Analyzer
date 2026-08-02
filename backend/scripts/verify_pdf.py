#!/usr/bin/env python
"""Full parser verification for XXXX6607.pdf using the production pdf_parser."""
import sys, logging
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pdf-parser"))

for lg in ["huggingface_hub", "sentence_transformers", "httpx", "urllib3",
           "transformers", "filelock"]:
    logging.getLogger(lg).setLevel(logging.ERROR)

from pdf_parser import parse_pdf

df = parse_pdf(r"C:\Users\rajak\Downloads\XXXX6607.pdf",
               output_dir=str(ROOT / "notebook" / "output"))

print("\n" + "=" * 62)
print("FULL PARSER OUTPUT — XXXX6607.pdf")
print("=" * 62)
print(f"  Rows extracted     : {len(df)}")
print(f"  Bank               : {df['Sender_Bank_Name'].iloc[0]}")
print(f"  Account Number     : {df['Sender_Account_Number'].iloc[0]}")
print(f"  Customer Name      : {df['Sender_Customer_Name'].iloc[0]}")
print(f"  Customer ID        : {df['Sender_Customer_ID'].iloc[0]}")
print(f"  Account Type       : {df['Sender_Account_Type'].iloc[0]}")
print(f"  IFSC               : {df['Sender_IFSC'].iloc[0]}")
print(f"  Currency           : {df['Currency'].iloc[0]}")
print(f"  Date range         : {df['Date'].dropna().min()}  →  {df['Date'].dropna().max()}")
amt = pd.to_numeric(df["Transaction_Amount"], errors="coerce").dropna()
print(f"  Amount range       : ₹{amt.min():,.2f}  →  ₹{amt.max():,.2f}")
print(f"  Total debits       : ₹{amt[amt<0].sum():,.2f}")
print(f"  Total credits      : ₹{amt[amt>0].sum():,.2f}")
print()
print("TRANSACTIONS:")
print(df[["Date", "Transaction_Mode", "Transaction_Amount"]].to_string())
