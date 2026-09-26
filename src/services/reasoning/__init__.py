"""
Agent 3 — Reasoning & Verification.

``analyze`` is the only entry point. It builds a deterministic brief from
Agent 2's rows, decides how soon the results should be acted on, asks the LLM
for readable prose, verifies every claim in that prose against the brief, and
scores its own confidence from observable signals.

The LLM is the only optional part. Everything else runs with no network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.services.reasoning import narrative as _narrative
from src.services.reasoning.brief import ClinicalBrief, build_brief
from src.services.reasoning.escalation import escalate
from src.services.reasoning.verify import VerificationReport, verify

logger = logging.getLogger(__name__)

__all__ = ["ReasoningResult", "analyze", "score_confidence"]


@dataclass
class ReasoningResult:
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    doctor_questions: List[str] = field(default_factory=list)
    lifestyle_tips: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    escalation_level: str = "routine"
    escalation_reasons: List[Dict[str, str]] = field(default_factory=list)
    degraded: bool = False


def score_confidence(brief: ClinicalBrief, verification: Any,
                     fell_back: bool) -> float:
    """
    How much to trust this analysis, from observable signals only.

    The model is never asked how sure it is. A degraded extraction, uncoded
    rows, rows Agent 2 could not evaluate, and dropped claims each pull the
    score down independently.
    """
    if brief.total_rows == 0:
        return 0.0

    score = 1.0
    score *= 0.5 if brief.extraction_degraded else 1.0
    score *= 0.6 + 0.4 * (brief.coded_rows / brief.total_rows)
    score *= 1.0 - 0.5 * (brief.unknown_rows / brief.total_rows)

    if fell_back:
        score *= 0.4
    elif not getattr(verification, "clean", True):
        score *= 0.7

    return round(max(0.0, min(1.0, score)), 2)


async def analyze(
    lab_results: Sequence[Dict[str, Any]],
    patient_info: Optional[Dict[str, Any]] = None,
    report_notes: Optional[Sequence[str]] = None,
    extraction_degraded: bool = False,
) -> ReasoningResult:
    """Run Agent 3 over one report's worth of Agent 2 rows."""
    brief = build_brief(lab_results, patient_info, report_notes, extraction_degraded)
    escalation = escalate(brief)

    reasons = [
        {"rule_id": r.rule_id, "test_name": r.test_name, "detail": r.detail}
        for r in escalation.reasons
    ]

    if brief.total_rows == 0:
        draft = _narrative.render_fallback(brief, escalation)
        return ReasoningResult(
            summary=draft.summary,
            confidence_score=0.0,
            escalation_level=escalation.level,
            escalation_reasons=reasons,
            degraded=True,
        )

    draft = None
    verification: Optional[VerificationReport] = None
    fell_back = False

    try:
        draft = await _narrative.narrate_llm(brief, escalation)
        verification = verify(draft, brief)

        if not verification.summary_ok:
            details = tuple(v.detail for v in verification.violations)
            logger.info("Agent 3 narrative rejected, retrying once: %s", details)
            draft = await _narrative.narrate_llm(brief, escalation, details)
            verification = verify(draft, brief)

        if not verification.summary_ok:
            logger.warning("Agent 3 narrative failed verification twice — "
                           "using template")
            draft = None
    except Exception as exc:
        logger.warning("Agent 3 LLM narrative unavailable (%s) — using template",
                       str(exc)[:200])
        draft = None

    if draft is None:
        fell_back = True
        draft = _narrative.render_fallback(brief, escalation)
        verification = verify(draft, brief)

    kept = verification.kept if verification else {}
    return ReasoningResult(
        summary=draft.summary,
        key_findings=kept.get("key_findings", list(draft.key_findings)),
        doctor_questions=kept.get("doctor_questions", list(draft.doctor_questions)),
        lifestyle_tips=kept.get("lifestyle_tips", list(draft.lifestyle_tips)),
        confidence_score=score_confidence(brief, verification, fell_back),
        escalation_level=escalation.level,
        escalation_reasons=reasons,
        degraded=fell_back or brief.extraction_degraded,
    )
