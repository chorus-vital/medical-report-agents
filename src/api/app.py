"""
FastAPI application with CORS, error handlers, and route registration.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import health, reports

# ── logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="Medical Report Analyzer",
    description=(
        "AI-powered medical lab report analysis — "
        "upload a PDF / image and get structured, flagged results."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow everything for localhost dev) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── routes ──
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])


# ── global error handler ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.getLogger("app").error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )
