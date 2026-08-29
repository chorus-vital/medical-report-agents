"""
FastAPI application with CORS, error handlers, and route registration.
"""

from __future__ import annotations

import logging

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

# ── Static UI / Webview Tester ──
static_dirs = [
    Path(__file__).resolve().parent.parent / "static",  # src/static
    Path(__file__).resolve().parent.parent.parent / "static",  # static
]
static_dir = next((d for d in static_dirs if d.exists()), static_dirs[0])
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_webview():
    """Serve the Agent 1 interactive testing webview."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse(
        "<h2>Medical Report Analyzer Webview</h2><p>Static UI not found.</p>"
    )


# ── global error handler ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.getLogger("app").error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )
