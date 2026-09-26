"""
Turning the brief into prose.

Two paths produce the same four fields. ``narrate_llm`` asks Gemini for
readable language; ``render_fallback`` builds flatter prose from templates with
no network call. The fallback is not a stub — it runs on every LLM failure and
in every offline test, so it is the guaranteed floor of the feature.
"""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field

from config.settings import settings
from src.services.reasoning.brief import ClinicalBrief
from src.services.reasoning.escalation import LEVEL_TEXT, Escalation

logger = logging.getLogger(__name__)


class NarrativeDraft(BaseModel):
    """The four narrative fields. Deliberately carries no confidence field."""

    summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    doctor_questions: List[str] = Field(default_factory=list)
    lifestyle_tips: List[str] = Field(default_factory=list)


def _describe(abnormality) -> str:
    unit = f" {abnormality.unit}" if abnormality.unit else ""
    interval = (f" (reference interval {abnormality.range_text})"
                if abnormality.range_text else "")
    return (f"{abnormality.test_name} is {abnormality.value}{unit}, "
            f"{abnormality.direction} the expected range{interval}")


def render_fallback(brief: ClinicalBrief, escalation: Escalation) -> NarrativeDraft:
    """Build the narrative from templates, with no LLM."""
    if brief.total_rows == 0:
        return NarrativeDraft(
            summary="No lab results could be read from this report, so there is "
                    "nothing to summarise.",
        )

    abnormal = len(brief.abnormalities)
    parts: List[str] = []
    if abnormal:
        parts.append(f"{abnormal} of {brief.total_rows} results are outside "
                     f"their reference interval.")
    else:
        parts.append(f"None of the {brief.total_rows} results are outside "
                     f"their reference interval.")

    for note in brief.lab_notes:
        parts.append(f'Your lab printed this note: "{note}"')

    parts.append(LEVEL_TEXT[escalation.level])

    if brief.extraction_degraded:
        parts.append("Some of this report could not be read automatically, so "
                     "the list above may be incomplete.")

    findings = [_describe(a) for a in brief.abnormalities]

    questions: List[str] = []
    if abnormal:
        names = ", ".join(a.test_name for a in brief.abnormalities[:3])
        questions.append(f"What could explain the results outside the range "
                         f"({names})?")
        questions.append("Do any of these results need to be repeated, and if "
                         "so, when?")
        questions.append("Do these results change anything about my current care?")

    tips: List[str] = []
    if abnormal:
        tips.append("Bring this report with you to your appointment.")
        tips.append("Write down any recent symptoms and when they started.")

    return NarrativeDraft(
        summary=" ".join(parts),
        key_findings=findings,
        doctor_questions=questions,
        lifestyle_tips=tips,
    )
