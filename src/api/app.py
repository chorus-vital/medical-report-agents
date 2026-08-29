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

from config.settings import settings
from src.api.routes import health, reports

# ── logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

# pdfminer logs a font-descriptor warning for every glyph in many real lab
# reports, which buries the extraction logs.
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def _check_model_config() -> None:
    """
    Warn loudly at startup when the model provider cannot possibly work.

    Without this the server starts happily, every upload silently degrades to
    the regex fallback, and the reports simply look unparseable.
    """
    provider = settings.MODEL_PROVIDER.lower()
    log = logging.getLogger("startup")

    if provider == "gemini" and not settings.GEMINI_API_KEY:
        log.error(
            "GEMINI_API_KEY is not set but MODEL_PROVIDER=gemini. "
            "AI extraction is DISABLED and uploads will fall back to a basic "
            "pattern matcher. Add GEMINI_API_KEY to your .env and restart."
        )
    elif provider == "groq" and not settings.GROQ_API_KEY:
        log.error(
            "GROQ_API_KEY is not set but MODEL_PROVIDER=groq. "
            "AI extraction is DISABLED. Add GROQ_API_KEY to your .env."
        )
    else:
        log.info(
            "Model provider: %s (%s)",
            provider,
            {
                "gemini": settings.GEMINI_MODEL,
                "ollama": settings.OLLAMA_MODEL,
                "groq": settings.GROQ_MODEL,
            }.get(provider, "unknown"),
        )


_check_model_config()

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
# Single source of truth. This used to fall back across two candidate
# directories that both held a copy of index.html, so edits to one of them
# silently had no effect on the served page.
static_dir = Path(__file__).resolve().parent.parent / "static"
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
