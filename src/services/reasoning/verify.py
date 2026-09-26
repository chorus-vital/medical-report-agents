"""
Check Agent 3's narrative against the brief it was written from.

What this catches: numbers that appear nowhere in the report, analytes that
were never tested, diagnosis language, and claims about rows Agent 2 could not
evaluate.

What this does not catch: an invented claim built from real numbers that names
no recognised analyte. The verifier bounds fabricated *facts*; it cannot bound
unsound *reasoning*. That limit is a large part of why Agent 3's scope stops at
facts and urgency — see the design spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from src.services.reasoning.brief import ClinicalBrief, normalise_number
from src.services.terminology import _alias_index, _load_ontology

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_NORMALITY = re.compile(r"\b(normal|abnormal|fine|healthy|within range|inside "
                        r"the range)\b", re.I)

# Verbs and vocabulary that turn an observation into a diagnosis.
_DIAGNOSIS_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\byou (?:have|likely have|may have|might have)\b",
    r"\bthis (?:indicates|means|suggests|confirms|points to)\b",
    r"\b(?:indicative|suggestive) of\b",
    r"\bcaused by\b",
    r"\bdiagnos(?:is|ed|tic)\b",
    r"\byou (?:are|appear) (?:anaemic|anemic|diabetic)\b",
    r"\b(?:anaemia|anemia|leukaemia|leukemia|leukopenia|neutropenia|thalassaemia|"
    r"thalassemia|infection|deficiency|cancer|sepsis|disease|disorder|syndrome)\b",
))

# Small integers and ordinals appear in ordinary prose ("one of your results").
_FREE_NUMBERS = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                           "100"})


@dataclass(frozen=True)
class Violation:
    kind: str     # "number" | "analyte" | "diagnosis" | "unknown_row"
    field: str
    text: str
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    clean: bool
    summary_ok: bool
    violations: Tuple[Violation, ...]
    kept: Dict[str, List[str]]


@lru_cache(maxsize=1)
def _alias_to_loinc() -> Dict[str, str]:
    """Every ontology alias mapped to its analyte's LOINC code."""
    ontology = _load_ontology()
    return {
        alias: ontology[key].loinc_code
        for alias, key in _alias_index().items()
        if ontology.get(key) and ontology[key].loinc_code
    }


def _strip_lab_notes(text: str, brief: ClinicalBrief) -> str:
    """
    Remove verbatim lab notes before the diagnosis scan.

    A lab's own note routinely contains exactly the vocabulary the blocklist
    bans ("rule out viral etiology in view of leukopenia"). Quoting it is the
    point; paraphrasing it is not. Only exact spans are exempt.
    """
    for note in brief.lab_notes:
        text = text.replace(note, " ")
    return text


def _check_numbers(text: str, field: str, brief: ClinicalBrief) -> List[Violation]:
    found: List[Violation] = []
    for raw in _NUMBER.findall(text):
        token = normalise_number(raw)
        if token in _FREE_NUMBERS or token in brief.allowed_numbers:
            continue
        found.append(Violation("number", field, text,
                               f"{raw} appears in no row of this report"))
    return found


def _check_analytes(text: str, field: str, brief: ClinicalBrief) -> List[Violation]:
    """
    Flag a recognised analyte this report did not carry.

    Matching is by LOINC, not by name: a report that printed "Hemoglobin" should
    still allow the word "haemoglobin", and both resolve to 718-7.
    """
    lowered = text.lower()
    alias_loinc = _alias_to_loinc()
    for alias, loinc in alias_loinc.items():
        if len(alias) < 5:
            continue
        if not re.search(rf"\b{re.escape(alias)}\b", lowered):
            continue
        if loinc in brief.allowed_loinc or alias in brief.allowed_analytes:
            continue
        return [Violation("analyte", field, text,
                          f"'{alias}' was not tested in this report")]
    return []


def _check_diagnosis(text: str, field: str, brief: ClinicalBrief) -> List[Violation]:
    scannable = _strip_lab_notes(text, brief)
    for pattern in _DIAGNOSIS_PATTERNS:
        match = pattern.search(scannable)
        if match:
            return [Violation("diagnosis", field, text,
                              f"diagnosis language: '{match.group(0)}'")]
    return []


def _check_unknown_rows(text: str, field: str, brief: ClinicalBrief) -> List[Violation]:
    if not _NORMALITY.search(text):
        return []
    lowered = text.lower()
    for name in brief.unknown_analytes:
        if len(name) < 5:
            continue
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return [Violation("unknown_row", field, text,
                              f"'{name}' was not evaluated, so it cannot be "
                              f"called normal or abnormal")]
    return []


def _check(text: str, field: str, brief: ClinicalBrief) -> List[Violation]:
    if not text or not text.strip():
        return []
    return (
        _check_numbers(text, field, brief)
        + _check_analytes(text, field, brief)
        + _check_diagnosis(text, field, brief)
        + _check_unknown_rows(text, field, brief)
    )


def verify(draft: Any, brief: ClinicalBrief) -> VerificationReport:
    """Check every claim in ``draft`` against ``brief``."""
    violations: List[Violation] = []
    kept: Dict[str, List[str]] = {}

    summary_violations = _check(getattr(draft, "summary", "") or "", "summary", brief)
    violations.extend(summary_violations)

    for field in ("key_findings", "doctor_questions", "lifestyle_tips"):
        surviving: List[str] = []
        for item in getattr(draft, field, None) or []:
            item_violations = _check(item, field, brief)
            if item_violations:
                violations.extend(item_violations)
            else:
                surviving.append(item)
        kept[field] = surviving

    return VerificationReport(
        clean=not violations,
        summary_ok=not summary_violations,
        violations=tuple(violations),
        kept=kept,
    )
