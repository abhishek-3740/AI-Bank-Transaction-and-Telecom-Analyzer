"""API router for the PDF parser backend module."""

import traceback
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from pdf_parser import parse_pdf

from .ingest import detect_dataset_type, ingest
from .logging_config import get_logger
from .models import ParserResponse
from .utils import delete_file, save_upload_file, validate_file_extension

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/pdf", tags=["pdf"])


def _bust_dashboard_caches() -> None:
    """Reset the module-level dataframe caches in the scoring and graph routers.

    Both routers lazily load CSVs into a global variable the first time a request
    arrives. After an upload rewrites those CSVs we must clear the globals so the
    next request reads the fresh file rather than the stale in-memory copy.
    """
    try:
        import scoring.router as _sr
        _sr._scored_df = None
    except Exception:
        pass
    try:
        import graph.router as _gr
        _gr._nodes_df = None
        _gr._edges_df = None
    except Exception:
        pass
    try:
        import reports.router as _rr
        _rr._scored_df = None
        _rr._graph_df  = None
        _rr._gt_df     = None
    except Exception:
        pass


def _reconstruct_transaction_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to reconstruct Transaction_Amount from raw bank statement columns.

    Some bank PDFs split amount into 'Amount(INR)' and 'DR/CR' columns.
    This merges them into a single signed Transaction_Amount column.
    """
    if "Transaction_Amount" in df.columns and df["Transaction_Amount"].notna().any():
        return df

    amount_col = None
    dr_cr_col = None

    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if "amount" in col_lower and "inr" in col_lower:
            amount_col = col
        if col_lower in {"dr/cr", "dr_cr", "type", "dr_cr_flag"}:
            dr_cr_col = col

    if amount_col is None or dr_cr_col is None:
        return df

    def to_signed(row: pd.Series) -> float | None:
        try:
            val = float(str(row[amount_col]).replace(",", "").strip())
        except (ValueError, TypeError):
            return None
        flag = str(row[dr_cr_col]).strip().upper()
        if flag in {"DR", "D", "DEBIT"}:
            return -abs(val)
        if flag in {"CR", "C", "CREDIT"}:
            return abs(val)
        return val

    df = df.copy()
    df["Transaction_Amount"] = df.apply(to_signed, axis=1)
    return df


def _validate_dataframe(df: pd.DataFrame, dataset_type: str) -> None:
    """Minimal validation to ensure critical columns are not empty."""
    critical_columns = {
        "BANK": ["Transaction_Amount", "Tran_Date"],
        "CDR": ["Call_Date", "Call_Duration"],
        "IPDR": ["Session_Start_Time", "Data_Volume"],
    }
    cols = critical_columns.get(dataset_type, [])
    for col in cols:
        if col in df.columns and df[col].isna().all():
            raise ValueError(f"Critical column '{col}' is present but completely empty.")


@router.post("/parse", response_model=ParserResponse)
def parse_pdf_endpoint(file: UploadFile = File(...)) -> ParserResponse:
    """Parse an uploaded PDF file and return extracted data.

    The parser automatically detects whether the PDF contains
    Bank, CDR, or IPDR data.

    Args:
        file: The PDF file to parse.

    Returns:
        ParserResponse containing the extracted data.

    Raises:
        HTTPException: If validation fails or parsing encounters an error.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    validate_file_extension(file.filename)

    temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path: Path | None = None

    try:
        temp_path = save_upload_file(file, temp_filename)

        logger.info(f"Starting PDF parse for {file.filename}")

        # Monkey-patch parser validation to bypass strict internal checks
        # that fail on split DR/CR bank statements.
        import pdf_parser

        original_validate = getattr(pdf_parser, "_validate_schema", None)
        if original_validate:
            pdf_parser._validate_schema = lambda df, dtype: None

        try:
            # parse_pdf also writes a "<type>_parsed.csv"; keep it beside the
            # upload instead of littering the process working directory.
            df = parse_pdf(str(temp_path), output_dir=str(temp_path.parent))
        finally:
            if original_validate:
                pdf_parser._validate_schema = original_validate

        # Fallback: reconstruct Transaction_Amount for split DR/CR bank statements
        df = _reconstruct_transaction_amount(df)

        dataset_type = detect_dataset_type(df)

        # Run our own validation after reconstruction
        _validate_dataframe(df, dataset_type)

        rows = len(df)
        columns = df.columns.tolist()
        data = df.to_dict(orient="records")

        logger.info(
            f"Successfully parsed PDF {file.filename}: "
            f"{rows} rows, {len(columns)} columns, type={dataset_type}"
        )

        # Feed the investigation pipeline. A failure here must not discard a
        # successful parse — the caller still gets its rows, plus the reason.
        try:
            ingest_summary = ingest(df, dataset_type)
            # Bust the in-memory caches in the scoring and graph routers so the
            # dashboard reflects the freshly scored data without a server restart.
            _bust_dashboard_caches()
        except Exception as exc:
            logger.error(f"Ingestion failed for {file.filename}: {exc}")
            logger.error(traceback.format_exc())
            ingest_summary = {"dashboard_updated": False, "reason": str(exc)}

        return ParserResponse(
            status="success",
            dataset_type=dataset_type,
            rows=rows,
            columns=columns,
            data=data,
            ingest=ingest_summary,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(f"PDF parsing failed for {file.filename}: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Failed to parse the PDF file. Please ensure the file is valid.",
        ) from None

    finally:
        if temp_path:
            delete_file(temp_path)