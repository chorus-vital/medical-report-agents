"""
Pydantic data models for lab results and report analysis.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LabResult(BaseModel):
    """A single lab test result with flag status."""

    test_name: str
    standard_name: Optional[str] = None
    loinc_code: Optional[str] = None
    observed_value: float | str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Literal["GREEN", "AMBER", "RED", "UNKNOWN"] = "UNKNOWN"
    explanation: Optional[str] = None


class ReportAnalysis(BaseModel):
    """Complete analysis output for a medical report."""

    id: str
    report_title: str
    patient_info: Optional[Dict[str, Any]] = None
    results: List[LabResult] = Field(default_factory=list)
    summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    doctor_questions: List[str] = Field(default_factory=list)
    lifestyle_tips: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    disclaimer: str = (
        "⚕️ This analysis is AI-generated for informational purposes only. "
        "It is NOT a medical diagnosis. Always consult a qualified healthcare "
        "professional for medical advice."
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
