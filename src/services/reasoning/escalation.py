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
    "no_data",
    "routine",
    "discuss_at_next_visit",
    "see_doctor_promptly",
    "seek_care_now",
)

LEVEL_TEXT = {
    "no_data": "No results could be read from this report.",
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
    # Which conversion table in _UNIT_FACTORS the row's declared unit is looked
    # up in. A rule with no family needs no conversion.
    quantity: Optional[str] = None


# Published critical and action thresholds, in the units named by ``describe``.
_THRESHOLDS: Tuple[_Threshold, ...] = (
    _Threshold("anc_critical", "751-8", "seek_care_now",
               lambda v: v < 500, "below 500 /mm3", quantity="cells"),
    _Threshold("platelet_critical", "777-3", "seek_care_now",
               lambda v: v < 20_000, "below 20,000 /mm3", quantity="cells"),
    _Threshold("hb_critical", "718-7", "seek_care_now",
               lambda v: v < 7, "below 7 g/dL", quantity="haemoglobin"),
    _Threshold("potassium_critical", "2823-3", "seek_care_now",
               lambda v: v < 2.5 or v > 6.5, "outside 2.5-6.5 mmol/L",
               quantity="potassium"),
    _Threshold("glucose_critical", "1558-6", "seek_care_now",
               lambda v: v < 50 or v > 500, "outside 50-500 mg/dL",
               quantity="glucose"),
    _Threshold("glucose_critical", "1521-4", "seek_care_now",
               lambda v: v < 50 or v > 500, "outside 50-500 mg/dL",
               quantity="glucose"),
    _Threshold("anc_moderate", "751-8", "see_doctor_promptly",
               lambda v: v < 1500, "below 1500 /mm3", quantity="cells"),
    _Threshold("platelet_low", "777-3", "see_doctor_promptly",
               lambda v: v < 50_000, "below 50,000 /mm3", quantity="cells"),
    _Threshold("hb_low", "718-7", "see_doctor_promptly",
               lambda v: v < 10, "below 10 g/dL", quantity="haemoglobin"),
)


# Multiply a value in the keyed unit to reach the rule's own unit. Declared
# units are auditable; inferring a scale from the magnitude of the printed range
# is not, and got both directions wrong — it fired a critical alert on a normal
# glucose in mmol/L and went silent on a platelet count of 15,000 whose range
# happened to be comma-grouped.
_UNIT_FACTORS = {
    # → /mm3 (identical to /uL)
    "cells": {
        "/mm3": 1, "cells/mm3": 1, "cells/cumm": 1, "/cumm": 1, "/ul": 1,
        "cells/ul": 1, "mill/mm3": 1_000_000, "million/ul": 1_000_000,
        "103/ul": 1000, "103/mm3": 1000, "k/ul": 1000, "thou/ul": 1000,
        "109/l": 1000,
        "lakhs/cumm": 100_000, "lakh/cumm": 100_000, "lakhs/mm3": 100_000,
        "lakhs/ul": 100_000,
    },
    # → g/dL
    "haemoglobin": {"g/dl": 1, "gm/dl": 1, "g%": 1, "gm%": 1, "g/l": 0.1},
    # → mmol/L
    "potassium": {"mmol/l": 1, "meq/l": 1},
    # → mg/dL
    "glucose": {"mg/dl": 1, "mg%": 1, "mmol/l": 18.0},
}

_SUPERSCRIPTS = str.maketrans({"³": "3", "⁹": "9", "µ": "u",
                               "μ": "u", "⁶": "6"})


def _normalise_unit(unit: Optional[str]) -> str:
    """Fold the many printed spellings of a unit onto one key."""
    if not unit:
        return ""
    text = unit.strip().lower().translate(_SUPERSCRIPTS)
    for junk in (" ", "^", "*", "x10", "×10"):
        text = text.replace(junk, "10" if junk in ("x10", "×10") else "")
    return text


def _as_float(text: str) -> Optional[float]:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _in_rule_units(value: float, unit: Optional[str],
                   rule: _Threshold) -> Optional[float]:
    """
    Convert ``value`` into the rule's units, or return None to skip the rule.

    Refusing to fire on an unrecognised unit is the safe direction: the row
    still carries its flag, so the severity rules below continue to apply.
    """
    if rule.quantity is None:
        return value
    factor = _UNIT_FACTORS[rule.quantity].get(_normalise_unit(unit))
    if factor is None:
        logger.debug("Skipping %s: unrecognised unit %r", rule.rule_id, unit)
        return None
    return value * factor


def escalate(brief: ClinicalBrief) -> Escalation:
    """Decide how soon these results should be acted on, and why."""
    if brief.total_rows == 0:
        # Absence of data is not absence of disease, and must not render as a
        # green "nothing outside range" banner.
        return Escalation(level="no_data", reasons=())

    reasons: List[FiredRule] = []

    already_fired: set = set()
    for abnormality in brief.abnormalities:
        if not abnormality.loinc_code:
            continue
        value = _as_float(abnormality.value)
        if value is None:
            continue
        for rule in _THRESHOLDS:
            if rule.loinc_code != abnormality.loinc_code:
                continue
            if (rule.rule_id, abnormality.loinc_code) in already_fired:
                continue  # the same analyte printed twice is still one finding
            try:
                converted = _in_rule_units(value, abnormality.unit, rule)
                fired = converted is not None and rule.test(converted)
            except Exception:  # a malformed threshold must not break the node
                logger.exception("Escalation rule %s raised", rule.rule_id)
                continue
            if fired:
                already_fired.add((rule.rule_id, abnormality.loinc_code))
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
