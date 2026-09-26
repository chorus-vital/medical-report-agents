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
_NORMALITY = re.compile(
    r"\b(normal|abnormal|fine|healthy|unremarkable|satisfactory|reassuring|"
    r"not (?:elevated|raised|low|high)|no concern|nothing to worry|"
    r"(?:with)?in (?:the )?(?:normal |reference |expected )?(?:range|interval|limits))\b",
    re.I)

# Phrasings that attribute a condition to this patient. Always blocked.
_DIAGNOSIS_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\byou (?:have|likely have|may have|might have)\b",
    r"\bthis (?:indicates|means|suggests|confirms|points to)\b",
    r"\b(?:indicative|suggestive) of\b",
    r"\bcaused by\b",
    r"\bdiagnos(?:is|ed|tic)\b",
    r"\byou (?:are|appear) (?:anaemic|anemic|diabetic)\b",
))

# Condition nouns. These are context-dependent: naming what a cell DOES is a
# definition ("white blood cells that fight infection"), while predicating the
# same noun of the patient is a diagnosis ("a sign of infection"). Blocking the
# bare noun dropped the two headline findings on a real report, because the
# model had reworded our own ontology gloss.
_CONDITION_NOUNS = re.compile(
    r"\b(anaemia|anemia|leukaemia|leukemia|leukopenia|neutropenia|thalassaemia|"
    r"thalassemia|infection|infections|deficiency|cancer|sepsis|disease|disorder|"
    r"syndrome)\b", re.I)

# Definitional use: the noun describes a cell's job, not the patient's state.
_DEFINITIONAL = re.compile(
    r"(?:fight|combat|defend|protect|guard|ward)\w*\s+(?:against\s+)?$"
    r"|(?:involved in|response to|responses to|protection against|defence against|"
    r"defense against)\s+$",
    re.I)
_COMPOUND = re.compile(r"^[-\s]?(?:fighting|fighter|related|linked)\b", re.I)

# Predication: the noun is being attached to this patient's results. "Findings
# suggest" is only a problem when a condition follows it — blocking the verb
# phrase outright rejected two Groq drafts in a row over
# "these findings suggest discussing the results with your doctor", and sent a
# real report to the template for no reason.
_PREDICATING = re.compile(
    r"\b(?:sign|signs|evidence|suggestive|indicative|consistent with|due to|"
    r"because of|likely|probably|possible|possibly|risk of|points? to|you have|"
    r"you may have|is an?|are an?|suffering from|suggests?|suggesting|"
    r"indicates?|indicating|shows?)\b[^.]{0,30}$", re.I)

# U+2011 and friends: the model reformats our gloss and the exact-span
# exemption stops matching, so dashes are folded before any of this runs.
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-",
                         "–": "-", "—": "-", "−": "-"})

# Numbers spelled as words, which the digit scan cannot see. Only decimals are
# rejected: "one of your results" is ordinary prose, "nine point two" is a value.
_WORD_DECIMAL = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred)\s+point\s+"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine)\b", re.I)

# Acronyms are matched case-sensitively as standalone uppercase tokens, so the
# analyte scan can reach TSH and ESR without firing on the word "alternative".
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,6}\b")

# The brief's own counts are already in allowed_numbers, so no free list is
# needed. Every digit in the narrative must trace to a row or a count.


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


# A note shorter than this is a heading or an OCR fragment, not a clinical
# remark. Exempting one would switch a blocklist term off for the whole
# narrative — report_notes comes from an LLM reading the PDF.
_MIN_EXEMPT_NOTE = 20


def _strip_lab_notes(text: str, brief: ClinicalBrief) -> str:
    """
    Remove quoted verbatim lab notes before the diagnosis scan.

    A lab's own note routinely contains exactly the vocabulary the blocklist
    bans ("rule out viral etiology in view of leukopenia"). Quoting it, with
    attribution, is the point. Replaying it unquoted as our own conclusion is
    not, so the span only earns its exemption inside quotation marks.
    """
    for gloss in brief.allowed_glosses:
        if len(gloss.strip()) >= _MIN_EXEMPT_NOTE:
            text = text.replace(gloss, " ")
    for note in brief.lab_notes:
        if len(note.strip()) < _MIN_EXEMPT_NOTE:
            continue
        for opening, closing in (('"', '"'), ("“", "”"), ("'", "'")):
            text = text.replace(f"{opening}{note}{closing}", " ")
    return text


def _check_numbers(text: str, field: str, brief: ClinicalBrief) -> List[Violation]:
    found: List[Violation] = []
    for raw in _NUMBER.findall(text):
        token = normalise_number(raw)
        if token in brief.allowed_numbers:
            continue
        found.append(Violation("number", field, text,
                               f"{raw} appears in no row of this report"))
    match = _WORD_DECIMAL.search(text)
    if match:
        found.append(Violation("number", field, text,
                               f"spelled-out value '{match.group(0)}' cannot be "
                               f"checked against the report"))
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
    scannable = _strip_lab_notes(text, brief).translate(_DASHES)

    for pattern in _DIAGNOSIS_PATTERNS:
        match = pattern.search(scannable)
        if match:
            return [Violation("diagnosis", field, text,
                              f"diagnosis language: '{match.group(0)}'")]

    for match in _CONDITION_NOUNS.finditer(scannable):
        before = scannable[max(0, match.start() - 40):match.start()]
        after = scannable[match.end():match.end() + 20]
        if _DEFINITIONAL.search(before) or _COMPOUND.match(after):
            continue  # describing what a cell does, not what the patient has
        if _PREDICATING.search(before) or not before.strip():
            return [Violation("diagnosis", field, text,
                              f"condition named: '{match.group(0)}'")]
        return [Violation("diagnosis", field, text,
                          f"condition named: '{match.group(0)}'")]
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
