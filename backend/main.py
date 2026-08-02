"""Main FastAPI application for the AI Telecom Bank Analyzer backend."""

import logging
import sys
from pathlib import Path

# Ensure the backend directory is on sys.path so that `pdf` resolves correctly
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Ensure pdf-parser/ is on sys.path so that `from pdf_parser import parse_pdf` works
_project_root = _backend_dir.parent
_pdf_parser_path = _project_root / "pdf-parser"
if str(_pdf_parser_path) not in sys.path:
    sys.path.insert(0, str(_pdf_parser_path))

from fastapi import FastAPI

from pdf.config import PROJECT_NAME, VERSION
from pdf.logging_config import get_logger
from pdf.router import router as pdf_router
from scoring.router import router as scoring_router
from graph.router import router as graph_router
from reports.router import router as reports_router

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = get_logger(__name__)

app = FastAPI(
    title=PROJECT_NAME,
    version="2.0.0",
    description=(
        "TRI-NETRA backend API — PDF parsing, investigation search, "
        "risk scoring, graph analytics, and STR reporting."
    ),
)

app.include_router(pdf_router)
app.include_router(scoring_router)
app.include_router(graph_router)
app.include_router(reports_router)


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "name": PROJECT_NAME,
        "version": "2.0.0",
        "status": "running",
        "endpoints": (
            "/api/v1/pdf/parse, "
            "/api/v1/scoring/alerts, "
            "/api/v1/scoring/transactions, "
            "/api/v1/scoring/customer/{id}, "
            "/api/v1/scoring/stats, "
            "/api/v1/scoring/score, "
            "/api/v1/graph/summary, "
            "/api/v1/graph/nodes, "
            "/api/v1/graph/node/{id}, "
            "/api/v1/graph/mules, "
            "/api/v1/graph/edges, "
            "/api/v1/reports/str/{customer_id}, "
            "/api/v1/reports/str/batch, "
            "/api/v1/reports/summary"
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}