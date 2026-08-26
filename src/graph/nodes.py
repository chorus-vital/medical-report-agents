"""
LangGraph agent nodes.

Currently implements:
  1. ``extract_node`` — Ingestion & Extraction agent

Upcoming:
  2. ``ground_node``  — Terminology & Flagging  (Agent 2)
  3. ``reason_and_verify_node`` — Reasoning & Verification  (Agent 3)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from src.schemas.state import PipelineState
from src.services.extractor import extract_from_file

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
        extracted_items, patient_info = await extract_from_file(
            file_path, file_type
        )
    except Exception as exc:
        logger.error("❌ [extract] Failed: %s", exc)
        return {
            "extracted_items": [],
            "patient_info": None,
            "report_id": report_id,
            "current_step": "extraction_failed",
            "errors": [f"Extraction error: {exc}"],
        }

    if not extracted_items:
        logger.warning("⚠️  [extract] No lab items found in document")
        return {
            "extracted_items": [],
            "patient_info": patient_info,
            "report_id": report_id,
            "current_step": "extraction_complete",
            "errors": ["No lab results could be extracted from the document."],
        }

    logger.info(
        "✅ [extract] Done — %d items, patient=%s",
        len(extracted_items),
        "yes" if patient_info else "no",
    )
    return {
        "extracted_items": extracted_items,
        "patient_info": patient_info,
        "report_id": report_id,
        "current_step": "extraction_complete",
        "errors": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node 2 — Terminology & Flagging  (placeholder)
# ═══════════════════════════════════════════════════════════════════════════

async def ground_node(state: PipelineState) -> Dict[str, Any]:
    """Placeholder — will be implemented as Agent 2."""
    logger.info("🏷️  [ground] Placeholder — passing through")
    return {"current_step": "grounding_complete"}


# ═══════════════════════════════════════════════════════════════════════════
# Node 3 — Reasoning & Verification  (placeholder)
# ═══════════════════════════════════════════════════════════════════════════

async def reason_and_verify_node(state: PipelineState) -> Dict[str, Any]:
    """Placeholder — will be implemented as Agent 3."""
    logger.info("🧠 [reason] Placeholder — passing through")
    return {"current_step": "reasoning_complete"}
