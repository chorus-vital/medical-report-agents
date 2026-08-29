"""
Report routes — upload, analyse, and retrieve medical reports.

Endpoints
---------
POST /api/reports/analyze          Upload + full pipeline (JSON response)
POST /api/reports/analyze/stream   Upload + SSE stream  (coming soon)
GET  /api/reports/history          List saved reports    (coming soon)
GET  /api/reports/{report_id}      Get a saved report    (coming soon)
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.settings import settings
from src.graph.pipeline import pipeline
from src.services.extractor import detect_file_type

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze")
async def analyze_report(file: UploadFile = File(...)):
    """
    Upload a medical report (PDF / image / docx / text) and receive a
    structured analysis with extracted lab items and patient info.

    The uploaded file is persisted under ``data/uploads/``.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # ── detect type early so we fail fast on unsupported formats ──
    try:
        file_type = detect_file_type(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── save the upload ──
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    save_path = upload_dir / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(
        "📂 Saved upload: %s (%s, %s)", save_path.name, file_type, ext
    )

    # ── run the pipeline ──
    initial_state = {
        "file_path": str(save_path),
        "file_type": file_type,
        "report_id": file_id,
    }

    try:
        result = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    errors = result.get("errors", [])
    extracted = result.get("extracted_items", [])

    return {
        "report_id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "status": "success" if extracted else "no_results",
        "patient_info": result.get("patient_info"),
        "extracted_items": extracted,
        "items_count": len(extracted),
        "errors": errors,
    }


@router.post("/analyze-text")
async def analyze_text_report(payload: dict):
    """
    Analyze raw lab report text directly.
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content is required.")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    save_path = upload_dir / f"{file_id}.txt"
    save_path.write_text(text, encoding="utf-8")

    initial_state = {
        "file_path": str(save_path),
        "file_type": "text",
        "raw_text": text,
        "report_id": file_id,
    }

    try:
        result = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline failed for pasted text: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    errors = result.get("errors", [])
    extracted = result.get("extracted_items", [])

    return {
        "report_id": file_id,
        "filename": "pasted_report.txt",
        "file_type": "text",
        "status": "success" if extracted else "no_results",
        "patient_info": result.get("patient_info"),
        "extracted_items": extracted,
        "items_count": len(extracted),
        "errors": errors,
    }


@router.get("/samples")
async def list_sample_reports():
    """
    List available bundled sample reports for quick testing.
    """
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        return {"samples": []}

    samples = []
    for p in sorted(samples_dir.iterdir()):
        if p.is_file():
            samples.append({
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "file_type": detect_file_type(p.name) if p.suffix else "unknown",
            })
    return {"samples": samples}


@router.post("/analyze-sample/{filename}")
async def analyze_sample_report(filename: str):
    """
    Run Agent 1 directly on a bundled sample report.
    """
    sample_path = Path("data/samples") / filename
    if not sample_path.exists() or not sample_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sample file '{filename}' not found.")

    try:
        file_type = detect_file_type(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    file_id = f"sample-{uuid.uuid4().hex[:8]}"

    initial_state = {
        "file_path": str(sample_path),
        "file_type": file_type,
        "report_id": file_id,
    }

    try:
        result = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline failed on sample: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    errors = result.get("errors", [])
    extracted = result.get("extracted_items", [])

    return {
        "report_id": file_id,
        "filename": filename,
        "file_type": file_type,
        "status": "success" if extracted else "no_results",
        "patient_info": result.get("patient_info"),
        "extracted_items": extracted,
        "items_count": len(extracted),
        "errors": errors,
    }
