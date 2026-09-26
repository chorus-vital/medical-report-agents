"""
How soon a result should be acted on.

This is the only clinical content in the codebase. Every threshold below is a
published critical value, every rule cites the row that fired it, and none of
them names a cause — they answer "how soon", never "why". Rules key on
``loinc_code`` so they can only fire on a row Agent 2 coded confidently; an
uncoded row lowers confidence instead of triggering an alert.

Units are the trap here. Labs print platelets as either 194 (x10^3/uL) or
194,000 (/mm3), and a threshold written for one convention fires constantly
under the other. Each rule that needs it carries a ``scale_threshold``: when
the printed reference range's largest bound falls below it, the value is in
thousands and is scaled before the comparison.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.services.reasoning.brief import ClinicalBrief

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

LEVELS: Tuple[str, ...] = (
    "routine",
    "discuss_at_next_visit",
    "see_doctor_promptly",
    "seek_care_now",
)

LEVEL_TEXT = {
    "routine": "Nothing here falls outside its reference range.",
    "discuss_at_next_visit": "Worth raising at your next routine appointment.",
    "see_doctor_promptly": "Worth showing to a doctor soon rather than waiting.",
    "seek_care_now": "Please contact a doctor today about these results.",
}

_AGGREGATE_LEVEL = {
    "any_red": "see_doctor_promptly",
    "any_amber": "discuss_at_next_visit",
}


@dataclass(frozen=True)
class FiredRule:
    rule_id: str
    test_name: str
    detail: str


@dataclass(frozen=True)
class Escalation:
    level: str
    reasons: Tuple[FiredRule, ...]


@dataclass(frozen=True)
class _Threshold:
    rule_id: str
    loinc_code: str
    level: str
    test: Callable[[float], bool]
    describe: str
    # When the printed range's largest bound is below this, the value is in
    # thousands and is multiplied by 1000 before ``test`` sees it.
    scale_threshold: Optional[float] = None


# Published critical and action thresholds, in the units named by ``describe``.
_THRESHOLDS: Tuple[_Threshold, ...] = (
    _Threshold("anc_critical", "751-8", "seek_care_now",
               lambda v: v < 500, "below 500 /mm3", scale_threshold=100),
    _Threshold("platelet_critical", "777-3", "seek_care_now",
               lambda v: v < 20_000, "below 20,000 /mm3", scale_threshold=10_000),
    _Threshold("hb_critical", "718-7", "seek_care_now",
               lambda v: v < 7, "below 7 g/dL"),
    _Threshold("potassium_critical", "2823-3", "seek_care_now",
               lambda v: v < 2.5 or v > 6.5, "outside 2.5-6.5 mmol/L"),
    _Threshold("glucose_critical", "1558-6", "seek_care_now",
               lambda v: v < 50 or v > 500, "outside 50-500 mg/dL"),
    _Threshold("glucose_critical", "1521-4", "seek_care_now",
               lambda v: v < 50 or v > 500, "outside 50-500 mg/dL"),
    _Threshold("anc_moderate", "751-8", "see_doctor_promptly",
               lambda v: v < 1500, "below 1500 /mm3", scale_threshold=100),
    _Threshold("platelet_low", "777-3", "see_doctor_promptly",
               lambda v: v < 50_000, "below 50,000 /mm3", scale_threshold=10_000),
    _Threshold("hb_low", "718-7", "see_doctor_promptly",
               lambda v: v < 10, "below 10 g/dL"),
)


def _as_float(text: str) -> Optional[float]:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _scaled(value: float, range_text: str, rule: _Threshold) -> float:
    """Bring a value printed in thousands up to the rule's own units."""
    if rule.scale_threshold is None:
        return value
    bounds = [float(n) for n in _NUMBER.findall(range_text or "")]
    if bounds and max(bounds) < rule.scale_threshold:
        return value * 1000
    return value


def escalate(brief: ClinicalBrief) -> Escalation:
    """Decide how soon these results should be acted on, and why."""
    reasons: List[FiredRule] = []

    for abnormality in brief.abnormalities:
        if not abnormality.loinc_code:
            continue
        value = _as_float(abnormality.value)
        if value is None:
            continue
        for rule in _THRESHOLDS:
            if rule.loinc_code != abnormality.loinc_code:
                continue
            try:
                fired = rule.test(_scaled(value, abnormality.range_text, rule))
            except Exception:  # a malformed threshold must not break the node
                logger.exception("Escalation rule %s raised", rule.rule_id)
                continue
            if fired:
                reasons.append(FiredRule(
                    rule_id=rule.rule_id,
                    test_name=abnormality.test_name,
                    detail=f"{abnormality.test_name} {abnormality.value} "
                           f"is {rule.describe}",
                ))

    most_severe = brief.abnormalities[0].test_name if brief.abnormalities else ""
    if brief.flag_counts.get("RED"):
        reasons.append(FiredRule("any_red", most_severe,
                                 f"{brief.flag_counts['RED']} result(s) outside "
                                 f"the reference interval"))
    elif brief.flag_counts.get("AMBER"):
        reasons.append(FiredRule("any_amber", most_severe,
                                 f"{brief.flag_counts['AMBER']} borderline result(s)"))

    level = "routine"
    for reason in reasons:
        rule_level = _AGGREGATE_LEVEL.get(reason.rule_id) or next(
            (r.level for r in _THRESHOLDS if r.rule_id == reason.rule_id), None
        )
        if rule_level and LEVELS.index(rule_level) > LEVELS.index(level):
            level = rule_level

    return Escalation(level=level, reasons=tuple(reasons))
