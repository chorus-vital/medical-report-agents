"""
LangGraph pipeline state definition.
"""

from typing import Any, Dict, List, Optional, TypedDict

from src.schemas.report import LabResult


class PipelineState(TypedDict, total=False):
    """Typed state flowing through the LangGraph pipeline."""

    # --- Input ---
    file_path: str
    file_type: str  # "pdf", "image", "text"
    raw_text: str

    # --- Extraction (Node 1 output) ---
    extracted_items: List[Dict[str, Any]]
    patient_info: Optional[Dict[str, Any]]

    # --- Terminology & Flagging (Node 2 output) ---
    lab_results: List[LabResult]

    # --- Reasoning & Verification (Node 3 output) ---
    summary: str
    key_findings: List[str]
    doctor_questions: List[str]
    lifestyle_tips: List[str]
    confidence_score: float

    # --- Metadata ---
    report_id: str
    errors: List[str]
    current_step: str
