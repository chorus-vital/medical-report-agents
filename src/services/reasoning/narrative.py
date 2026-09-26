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


_SYSTEM_RULES = """\
You are writing a plain-language explanation of one lab report for the patient
who took it. Follow these rules exactly.

1. Use only the facts in the BRIEF below. Never state a number that does not
   appear there.
2. Never diagnose. Do not name a disease, condition, or cause. Do not write
   "anaemia", "infection", "deficiency", or any similar term. Do not write
   "this indicates", "this suggests", "this means", or "you have".
3. Describe what a value is and whether it is inside or outside its range.
   Nothing more.
4. Never describe a result the brief does not list as outside its range, and
   say nothing at all about whether a NOT EVALUATED result is normal.
5. If the brief has lab notes, you may quote one word-for-word inside quotation
   marks. Never paraphrase a lab note.
6. Write at about an eighth-grade reading level. Short sentences, second
   person, no jargon without a plain-language gloss.
7. doctor_questions are questions the patient should ask their doctor.
   lifestyle_tips are practical, non-medical preparation steps. Never suggest a
   treatment, supplement, medication, or dose.
"""


def build_prompt(brief: ClinicalBrief, escalation: Escalation,
                 violations: tuple = ()) -> str:
    """The full prompt: rules, the rendered brief, and any prior violations."""
    from src.services.reasoning.brief import render_brief

    sections = [
        _SYSTEM_RULES,
        "",
        "BRIEF",
        "-----",
        render_brief(brief),
        "",
        f"HOW SOON TO ACT: {LEVEL_TEXT[escalation.level]}",
    ]
    if violations:
        sections += [
            "",
            "YOUR PREVIOUS ATTEMPT WAS REJECTED. Fix each of these and do not",
            "repeat them:",
        ]
        sections += [f"  - {v}" for v in violations]
    return "\n".join(sections)


def is_usable(draft: NarrativeDraft) -> bool:
    """
    Structured output guarantees shape, not content.

    A reply with an empty summary is schema-valid and useless, so it is
    treated as a failure and handed to the fallback.
    """
    return bool(draft.summary and draft.summary.strip())


async def narrate_llm(brief: ClinicalBrief, escalation: Escalation,
                      violations: tuple = ()) -> NarrativeDraft:
    """
    Ask Gemini for the narrative. Raises on any failure.

    Reuses the extractor's client factory and retry helper rather than
    introducing a second way of talking to the same API.
    """
    from google.genai import types as genai_types

    from src.services.extractor import _with_retries
    from src.services.llm_factory import get_vision_model

    client = get_vision_model()
    prompt = build_prompt(brief, escalation, violations)

    config = genai_types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=NarrativeDraft,
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )

    async def call():
        return await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

    response = await _with_retries(call, "Agent 3 narrative")

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, NarrativeDraft):
        draft = parsed
    elif isinstance(parsed, dict):
        draft = NarrativeDraft.model_validate(parsed)
    else:
        draft = NarrativeDraft.model_validate_json(getattr(response, "text", "") or "{}")

    if not is_usable(draft):
        raise ValueError("LLM returned an empty narrative")
    return draft
