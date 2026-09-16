"""
LangGraph agent nodes.

Currently implements:
  1. ``extract_node`` — Ingestion & Extraction agent
  2. ``ground_node``  — Terminology & Flagging agent

Upcoming:
  3. ``reason_and_verify_node`` — Reasoning & Verification  (Agent 3)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from src.schemas.state import PipelineState
from src.services.extractor import extract_from_file_detailed
from src.services.terminology import annotate_items

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Node 1 — Ingestion & Extraction
# ═══════════════════════════════════════════════════════════════════════════

async def extract_node(state: PipelineState) -> Dict[str, Any]:
    """
    **Agent 1 · Ingestion & Extraction**

    Accepts a PDF / image / text file, runs document parsing (pdfplumber
    or Gemini Vision), then uses an LLM structured prompt to produce a
    list of extracted lab items and patient demographics.

    State consumed
    ──────────────
    ``file_path``, ``file_type``

    State produced
    ──────────────
    ``extracted_items``, ``patient_info``, ``raw_text`` (debug),
    ``report_id``, ``current_step``, ``errors``
    """
    file_path: str = state["file_path"]
    file_type: str = state["file_type"]
    report_id: str = state.get("report_id") or str(uuid.uuid4())

    logger.info(
        "🔬 [extract] Starting — file=%s  type=%s  id=%s",
        file_path,
        file_type,
        report_id,
    )

    try:
        result = await extract_from_file_detailed(file_path, file_type)
    except Exception as exc:
        logger.error("❌ [extract] Failed: %s", exc, exc_info=True)
        return {
            "extracted_items": [],
            "patient_info": None,
            "report_notes": [],
            "extraction_method": "failed",
            "extraction_degraded": True,
            "warnings": [],
            "report_id": report_id,
            "current_step": "extraction_failed",
            "errors": [f"Extraction error: {exc}"],
        }

    errors: list[str] = []
    if not result.items:
        logger.warning("⚠️  [extract] No lab items found in document")
        errors.append("No lab results could be extracted from the document.")

    if result.degraded:
        # Surfaced to the caller so a fallback run is never presented as a
        # clean one — this is exactly the failure mode that made a
        # misconfigured API key look like a bad document.
        logger.warning("⚠️  [extract] Degraded extraction via %s", result.method)

    logger.info(
        "✅ [extract] Done — %d items via %s, patient=%s",
        len(result.items),
        result.method,
        "yes" if result.patient_info else "no",
    )
    return {
        "extracted_items": result.items,
        "patient_info": result.patient_info,
        "report_notes": result.notes,
        "extraction_method": result.method,
        "extraction_degraded": result.degraded,
        "warnings": result.warnings,
        "report_id": report_id,
        "current_step": "extraction_complete",
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — Terminology & Flagging
# ═══════════════════════════════════════════════════════════════════════════

async def ground_node(state: PipelineState) -> Dict[str, Any]:
    """
    **Agent 2 · Terminology & Flagging**

    Resolves each row Agent 1 extracted to a LOINC concept via fuzzy
    matching against the local ontology (``data/lab_ontology.json``), then
    evaluates the observed value against a reference interval to assign a
    GREEN / AMBER / RED status. The report's own printed interval always
    wins over the ontology's demographic-specific default; a test name that
    cannot be confidently matched is left uncoded rather than mis-coded.

    State consumed
    ──────────────
    ``extracted_items``, ``patient_info``

    State produced
    ──────────────
    ``lab_results``, ``current_step``
    """
    items = state.get("extracted_items") or []
    patient_info = state.get("patient_info")

    if not items:
        logger.info("🏷️  [ground] No extracted items to ground — passing through")
        return {"lab_results": [], "current_step": "grounding_complete"}

    results = annotate_items(items, patient_info)

    coded = sum(1 for r in results if r.get("loinc_code"))
    flag_counts = {f: sum(1 for r in results if r["flag"] == f) for f in ("GREEN", "AMBER", "RED", "UNKNOWN")}
    logger.info(
        "🏷️  [ground] Done — %d/%d rows matched to LOINC, flags=%s",
        coded, len(results), flag_counts,
    )
    return {"lab_results": results, "current_step": "grounding_complete"}


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — Reasoning & Verification  (placeholder)
# ═══════════════════════════════════════════════════════════════════════════

async def reason_and_verify_node(state: PipelineState) -> Dict[str, Any]:
    """Placeholder — will be implemented as Agent 3."""
    logger.info("🧠 [reason] Placeholder — passing through")
    return {"current_step": "reasoning_complete"}
