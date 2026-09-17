"""
Local LOINC matcher & reference range evaluator.
Uses data/lab_ontology.json for fuzzy matching and flagging.

Matching principle
-------------------
A wrong LOINC code is worse than no code. Each candidate test name is scored
against the ontology with rapidfuzz's WRatio; if nothing clears
MATCH_THRESHOLD, the row is deliberately left uncoded (loinc_code=None)
rather than guessing.

Beyond GREEN/AMBER/RED
-----------------------
A small set of clinically critical tests (Hemoglobin, Potassium, Sodium,
Glucose, WBC, Platelets) carry a "panic value" threshold in the ontology --
a result beyond that threshold is dangerous enough that a lab would phone
the ordering doctor immediately, which plain RED does not communicate. The
LabResult schema's `flag` field is fixed to GREEN/AMBER/RED/UNKNOWN, so a
critical result stays flagged RED but its `explanation` is prefixed with an
explicit escalation notice rather than the routine "above/below range" text.

Unit-mismatch awareness
------------------------
If the unit printed on the report doesn't match the ontology's expected unit
for that test (e.g. a glucose reported in mmol/L against a mg/dL reference
range), a flag comparison against the wrong scale is worse than no
comparison. compute_flag() detects this and returns UNKNOWN with an
explanation instead of silently producing a wrong verdict.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz, process

from src.schemas.report import LabResult

logger = logging.getLogger(__name__)

_ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "data" / "lab_ontology.json"

# Confidence (0-100) a fuzzy match must clear to be trusted. Below this, the
# row is left uncoded rather than risk attaching the wrong LOINC code.
MATCH_THRESHOLD = 80

# A near-miss band below MATCH_THRESHOLD: a match here is too uncertain to
# code automatically, but too close to be silently discarded either. These
# rows are surfaced as "needs review" with the closest candidate named, so a
# human can confirm in one glance instead of the row vanishing with no trace.
REVIEW_THRESHOLD = 65

_PAREN_RE = re.compile(r"\(([^)]+)\)")

# Units that are interchangeable for matching purposes (symbol variants of
# the same physical unit). Anything not in the same group is treated as a
# genuine mismatch worth flagging rather than silently comparing.
_UNIT_EQUIVALENCE_GROUPS = [
    {"mg/dl", "mg/dL".lower()},
    {"g/dl", "g/dl"},
    {"u/l", "iu/l"},
    {"%"},
    {"mmol/l"},
    {"meq/l", "mmol/l"},  # for monovalent ions these are numerically equal
    {"pg/ml"},
    {"ng/ml"},
    {"ng/dl"},
    {"uiu/ml", "miu/l", "µiu/ml"},
    {"mill/mm3", "million/µl", "million/ul", "10^6/ul"},
    {"cells/mm3", "/mm3", "/ul", "cells/µl"},
    {"10^3/ul", "10\u00b3/\u00b5l", "k/ul"},
    {"fl"},
    {"pg"},
]


def _canon_unit(unit: Optional[str]) -> str:
    if not unit:
        return ""
    u = unit.strip().lower()
    u = u.replace("μ", "u").replace("µ", "u")
    # Normalise Unicode superscript digits (³, ², ¹) to plain ASCII digits --
    # "cells/mm³" and "cells/mm3" are the same unit, and comparing them
    # without this step produces a false unit-mismatch on any report that
    # uses the superscript form, which most lab PDFs do.
    u = u.translate(str.maketrans("¹²³⁴⁵⁶⁷⁸⁹⁰", "1234567890"))
    u = re.sub(r"\s+", "", u)
    return u


def _units_compatible(reported: Optional[str], expected: Optional[str]) -> bool:
    """True if two unit strings are the same or known-equivalent; also true
    when either side is missing (nothing to contradict)."""
    r, e = _canon_unit(reported), _canon_unit(expected)
    if not r or not e:
        return True
    if r == e:
        return True
    for group in _UNIT_EQUIVALENCE_GROUPS:
        canon_group = {_canon_unit(g) for g in group}
        if r in canon_group and e in canon_group:
            return True
    return False


# ─────────────────────────── Ontology loading ────────────────────────────────

_ontology_cache: Optional[List[Dict[str, Any]]] = None
_alias_index_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _load_ontology() -> List[Dict[str, Any]]:
    """Load and cache the lab ontology JSON."""
    global _ontology_cache
    if _ontology_cache is not None:
        return _ontology_cache

    try:
        with open(_ONTOLOGY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _ontology_cache = data.get("tests", [])
        logger.info("Loaded %d ontology entries from %s", len(_ontology_cache), _ONTOLOGY_PATH)
    except Exception as exc:
        logger.error("Failed to load lab ontology: %s", exc)
        _ontology_cache = []

    return _ontology_cache


def _alias_index() -> Dict[str, Dict[str, Any]]:
    """Build (and cache) a flat map of {alias_text: ontology_entry}."""
    global _alias_index_cache
    if _alias_index_cache is not None:
        return _alias_index_cache

    index: Dict[str, Dict[str, Any]] = {}
    for entry in _load_ontology():
        names = [entry.get("standard_name", "")] + entry.get("aliases", [])
        for name in names:
            if name:
                index[name] = entry

    _alias_index_cache = index
    return index


def _name_variants(test_name: str) -> List[str]:
    """
    Produce the variants of a printed test name worth trying:
      1. the full name as printed
      2. any abbreviation found in parentheses, e.g. "(MCV)" -> "MCV"
      3. the name with the parenthetical stripped out
    """
    variants = [test_name]

    for match in _PAREN_RE.finditer(test_name):
        abbrev = match.group(1).strip()
        if abbrev:
            variants.append(abbrev)

    stripped = _PAREN_RE.sub("", test_name).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    if stripped and stripped not in variants:
        variants.append(stripped)

    return variants


def match_test(test_name: str) -> tuple[Optional[Dict[str, Any]], float, List[tuple[str, float]]]:
    """
    Find the best-matching ontology entry for an extracted test name.

    Returns (entry_or_None, confidence_0_to_100, top_candidates). Tries the
    full name, any parenthetical abbreviation, and the name with the
    parenthetical stripped, scoring each against every alias with
    rapidfuzz's WRatio. entry is None if nothing clears MATCH_THRESHOLD --
    an uncoded row is safer than a wrongly coded one. top_candidates lists
    up to 3 (standard_name, score) pairs regardless of threshold, so a
    near-miss can still name what it almost matched.
    """
    if not test_name:
        return None, 0.0, []

    index = _alias_index()
    if not index:
        return None, 0.0, []

    choices = list(index.keys())

    # Track the best score seen per distinct ontology entry (an entry can be
    # reached via several aliases/variants), so candidates aren't just the
    # same test listed three times under different alias spellings.
    best_per_entry: Dict[int, tuple[Dict[str, Any], float]] = {}

    for variant in _name_variants(test_name):
        results = process.extract(variant, choices, scorer=fuzz.WRatio, limit=5)
        for matched_string, score, _ in results:
            entry = index[matched_string]
            key = id(entry)
            if key not in best_per_entry or score > best_per_entry[key][1]:
                best_per_entry[key] = (entry, score)

    ranked = sorted(best_per_entry.values(), key=lambda pair: pair[1], reverse=True)
    top_candidates = [(e.get("standard_name", "?"), s) for e, s in ranked[:3]]

    if not ranked:
        return None, 0.0, []

    best_entry, best_score = ranked[0]

    if best_score >= MATCH_THRESHOLD:
        return best_entry, best_score, top_candidates

    logger.debug("No confident ontology match for %r (best score %.1f)", test_name, best_score)
    return None, best_score, top_candidates


# ─────────────────────────── Range parsing & flagging ─────────────────────────


def _parse_numeric(value: Any) -> Optional[float]:
    """Try to pull a float out of an observed value ('15.3', '<0.5', '15.3 H')."""
    if value is None:
        return None
    match = re.search(r"-?\d+\.?\d*", str(value))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _parse_range(range_str: Optional[str]) -> Optional[tuple[float, float]]:
    """
    Parse a reference range string into (low, high).

    Handles two-sided ('13 - 17'), one-sided ('<200', '>=90', 'up to 40',
    'X and above'), and returns None for qualitative ranges so callers fall
    back to UNKNOWN rather than mis-parsing them as numeric.
    """
    if not range_str:
        return None
    range_str = range_str.strip()

    m = re.match(r"^[<]=?\s*([\d.]+)$", range_str)
    if m:
        return (float("-inf"), float(m.group(1)))

    m = re.match(r"^[>]=?\s*([\d.]+)$", range_str)
    if m:
        return (float(m.group(1)), float("inf"))

    m = re.match(r"^up to\s+([\d.]+)$", range_str, re.I)
    if m:
        return (float("-inf"), float(m.group(1)))

    m = re.match(r"^([\d.]+)\s*(?:and above|or (?:more|higher|greater))$", range_str, re.I)
    if m:
        return (float(m.group(1)), float("inf"))

    m = re.match(r"^([\d.]+)\s*[-–—]\s*([\d.]+)$", range_str)
    if m:
        return (float(m.group(1)), float(m.group(2)))

    return None


_QUALITATIVE_NORMAL = {"nil", "not seen", "negative", "none seen", "absent", "trace"}
_QUALITATIVE_ABNORMAL = {"positive", "present", "seen", "1+", "2+", "3+", "4+"}


def _evaluate_qualitative(observed_value: Any, reference_range: Optional[str]) -> Optional[tuple[str, str]]:
    """Handle qualitative results ('Nil', 'Positive', 'Not seen') by text match."""
    if observed_value is None:
        return None
    text = str(observed_value).strip().lower()

    if text in _QUALITATIVE_NORMAL:
        return "GREEN", "Within normal (qualitative) expectation."
    if text in _QUALITATIVE_ABNORMAL:
        if reference_range and str(reference_range).strip().lower() in _QUALITATIVE_NORMAL:
            return "RED", "Qualitative result is abnormal for this test."
        return "UNKNOWN", None

    return None


def compute_flag(
    observed_value: Any,
    reference_range: Optional[str],
    reported_unit: Optional[str] = None,
    expected_unit: Optional[str] = None,
    critical: Optional[Dict[str, float]] = None,
) -> tuple[str, Optional[str]]:
    """
    Compare an observed value against a reference range.

    Returns (flag, explanation). Flag is one of GREEN/AMBER/RED/UNKNOWN.
    AMBER marks borderline results (within 10% of a limit); RED marks
    anything clearly outside range. If the result crosses a critical/panic
    threshold, the flag stays RED (the schema has no CRITICAL tier) but the
    explanation is prefixed with an explicit escalation notice. A unit
    mismatch between the report and the ontology short-circuits to UNKNOWN
    rather than risk comparing on the wrong scale.
    """
    if not _units_compatible(reported_unit, expected_unit):
        return "UNKNOWN", (
            f"Unit mismatch: report gives '{reported_unit}', ontology expects "
            f"'{expected_unit}' — not compared to avoid a wrong verdict."
        )

    qualitative = _evaluate_qualitative(observed_value, reference_range)
    if qualitative is not None:
        return qualitative

    numeric_value = _parse_numeric(observed_value)
    bounds = _parse_range(reference_range)

    if numeric_value is None or bounds is None:
        return "UNKNOWN", None

    low, high = bounds

    # Critical/panic threshold check -- takes priority over the routine
    # range verdict when breached, since it changes what the result means.
    if critical:
        crit_low = critical.get("low")
        crit_high = critical.get("high")
        if crit_low is not None and numeric_value <= crit_low:
            return "RED", f"🚨 CRITICAL LOW ({numeric_value} ≤ panic threshold {crit_low}) — outside normal range ({reference_range})."
        if crit_high is not None and numeric_value >= crit_high:
            return "RED", f"🚨 CRITICAL HIGH ({numeric_value} ≥ panic threshold {crit_high}) — outside normal range ({reference_range})."

    span = (high - low) if (high != float("inf") and low != float("-inf")) else None
    margin = span * 0.1 if span else 0.0

    if low <= numeric_value <= high:
        if span and (numeric_value - low < margin or high - numeric_value < margin):
            return "AMBER", f"Within range but close to the limit ({reference_range})."
        return "GREEN", "Within normal range."

    if numeric_value < low:
        return "RED", f"Below the normal range ({reference_range})."
    return "RED", f"Above the normal range ({reference_range})."


# ─────────────────────────── Public API ───────────────────────────────────────


def _ontology_default_range(entry: Dict[str, Any], sex: Optional[str]) -> Optional[str]:
    """Pick the sex-specific reference range from an ontology entry if present."""
    ranges = entry.get("reference_range", {}) or {}
    if sex:
        sex_key = sex.strip().lower()
        if sex_key in ("male", "m") and ranges.get("male"):
            return ranges["male"]
        if sex_key in ("female", "f") and ranges.get("female"):
            return ranges["female"]
    return ranges.get("default")


def ground_lab_item(item: Dict[str, Any], patient_sex: Optional[str] = None) -> LabResult:
    """
    Convert one extracted lab item into a grounded, flagged LabResult.

    Parameters
    ----------
    item : dict
        Shape produced by Agent 1 (src/services/extractor.py), e.g.:
        {"test_name": "Hemoglobin (Hb)", "observed_value": "15.3",
         "unit": "g/dL", "reference_range": "13 - 17", ...}
    patient_sex : str, optional
        "Male" / "Female", used to pick a sex-specific ontology range when
        the report itself doesn't print one.
    """
    test_name = item.get("test_name", "")
    observed_value_raw = item.get("observed_value")
    reported_range = item.get("reference_range")
    reported_unit = item.get("unit")

    entry, match_score, top_candidates = match_test(test_name)

    standard_name = entry.get("standard_name") if entry else None
    loinc_code = entry.get("loinc_code") if entry else None
    expected_unit = entry.get("unit") if entry else None
    unit = reported_unit or expected_unit
    critical = entry.get("critical") if entry else None

    # The report's own printed range always wins; the ontology default
    # (which may be sex-specific) is only a fallback for a missing range.
    reference_range = reported_range
    if not reference_range and entry:
        reference_range = _ontology_default_range(entry, patient_sex)

    numeric_value = _parse_numeric(observed_value_raw)
    observed_value = numeric_value if numeric_value is not None else str(observed_value_raw)

    flag, explanation = compute_flag(
        observed_value_raw,
        reference_range,
        reported_unit=reported_unit,
        expected_unit=expected_unit,
        critical=critical,
    )

    # Below MATCH_THRESHOLD the row stays uncoded, but a near-miss in the
    # REVIEW_THRESHOLD band is worth naming rather than vanishing silently --
    # a reviewer can confirm "Total RBC Count" was meant to be "RBC Count"
    # in one glance instead of re-deriving it from the raw report.
    if entry is None:
        if REVIEW_THRESHOLD <= match_score < MATCH_THRESHOLD and top_candidates:
            candidate_name, candidate_score = top_candidates[0]
            review_note = (
                f"🔎 Needs review: closest ontology match is '{candidate_name}' "
                f"(confidence {candidate_score:.0f}/100, below the {MATCH_THRESHOLD} "
                f"auto-accept threshold). Not auto-coded."
            )
            explanation = f"{review_note} {explanation}".strip() if explanation else review_note
        elif flag == "UNKNOWN" and explanation is None:
            explanation = f"No confident match found in the terminology ontology (best score {match_score:.0f}/100)."

    return LabResult(
        test_name=test_name,
        standard_name=standard_name,
        loinc_code=loinc_code,
        observed_value=observed_value,
        unit=unit,
        reference_range=reference_range,
        flag=flag,
        explanation=explanation,
    )


def ground_lab_items(
    items: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None
) -> List[LabResult]:
    """Ground and flag a full list of extracted lab items."""
    patient_sex = (patient_info or {}).get("sex")
    results = [ground_lab_item(item, patient_sex) for item in items]
    logger.info(
        "Grounded %d item(s): %d matched, %d unmatched, %d critical",
        len(results),
        sum(1 for r in results if r.loinc_code),
        sum(1 for r in results if not r.loinc_code),
        sum(1 for r in results if r.explanation and "CRITICAL" in r.explanation),
    )
    return results