"""
Terminology & Flagging (Agent 2).

Local LOINC matcher & reference-range evaluator, backed by
``data/lab_ontology.json``.

Design (mirrors the Review-1 report, ch. 4.3 "Module 2"):
  * Identity comes from fuzzy matching the printed test name — plus any
    parenthesised abbreviation — against the ontology's alias set. A match
    below ``FUZZY_MATCH_THRESHOLD`` is left uncoded rather than forced,
    because a wrong LOINC code is worse than no code.
  * Status comes from evaluating the observed value against a reference
    interval that may be two-sided ("13 - 17"), one-sided ("< 200",
    "> 90"), or qualitative ("Nil", "Positive"). The report's own printed
    interval always wins over the ontology default, since reference
    intervals are method- and instrument-dependent and the issuing
    laboratory is the authority on its own assay. The ontology default
    (optionally sex-specific) is used only when the report prints none.
  * The three-level output distinguishes a value inside the interval
    (GREEN), a value marginally outside it (AMBER), and a value far
    enough outside to warrant prompt attention (RED).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz, process

from config.settings import settings

logger = logging.getLogger(__name__)

# Below this rapidfuzz score (0-100), a row is left uncoded rather than
# risking a wrong LOINC code on a confident-looking but wrong match.
FUZZY_MATCH_THRESHOLD = 80.0

# How far outside a boundary still counts as "borderline" (AMBER) rather
# than "alert" (RED) — expressed as a fraction of the interval's width
# (or, for a one-sided bound, of the bound's own magnitude).
AMBER_BAND_PCT = 0.10


# ─────────────────────────────── Ontology ────────────────────────────────────


@dataclass
class OntologyEntry:
    key: str
    loinc_code: Optional[str]
    standard_name: str
    aliases: List[str]
    unit: Optional[str]
    panel: Optional[str]
    kind: str  # "numeric" | "qualitative"
    reference: Dict[str, Any]
    normal_values: List[str] = field(default_factory=list)
    explanation: str = ""


def _ontology_path() -> Path:
    path = Path(settings.LAB_ONTOLOGY_PATH)
    if not path.is_absolute():
        path = settings.BASE_DIR / path
    return path


@lru_cache(maxsize=1)
def _load_ontology() -> Dict[str, OntologyEntry]:
    """Load and index ``data/lab_ontology.json``. Cached for the process lifetime."""
    path = _ontology_path()
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.error("Lab ontology file not found at %s — Agent 2 will leave every row uncoded", path)
        return {}

    entries: Dict[str, OntologyEntry] = {}
    for item in raw.get("tests", []):
        key = item.get("key")
        if not key:
            continue
        aliases = {a.lower().strip() for a in item.get("aliases", [])}
        aliases.add(key.lower())
        aliases.add(item.get("standard_name", key).lower())
        entries[key] = OntologyEntry(
            key=key,
            loinc_code=item.get("loinc_code"),
            standard_name=item.get("standard_name", key),
            aliases=sorted(aliases),
            unit=item.get("unit"),
            panel=item.get("panel"),
            kind=item.get("kind", "numeric"),
            reference=item.get("reference", {}) or {},
            normal_values=[v.lower() for v in item.get("normal_values", [])],
            explanation=item.get("explanation", ""),
        )
    logger.info("Loaded lab ontology: %d analytes from %s", len(entries), path)
    return entries


def reload_ontology() -> None:
    """Drop the cached ontology so the next lookup re-reads the JSON file (tests/hot-reload)."""
    _load_ontology.cache_clear()


@lru_cache(maxsize=1)
def _alias_index() -> Dict[str, str]:
    """Every lowercase alias string -> ontology key, forming the fuzzy-match corpus."""
    index: Dict[str, str] = {}
    for key, entry in _load_ontology().items():
        for alias in entry.aliases:
            index.setdefault(alias, key)
    return index


_PAREN = re.compile(r"\(([^)]+)\)")


def _candidate_terms(raw_name: str) -> List[str]:
    """
    Strings worth trying against the alias index: the full printed name, any
    parenthesised abbreviation inside it (e.g. the "MCV" in "Mean Corpuscular
    Volume (MCV)"), and the name with the parenthetical stripped out.
    """
    name = (raw_name or "").strip().lower()
    if not name:
        return []
    terms = [name]
    for m in _PAREN.finditer(raw_name or ""):
        abbrev = m.group(1).strip().lower()
        if abbrev:
            terms.append(abbrev)
    stripped = _PAREN.sub("", raw_name or "").strip().lower()
    if stripped and stripped not in terms:
        terms.append(stripped)
    return terms


def match_to_loinc(test_name: str) -> Optional[Dict[str, Any]]:
    """
    Fuzzy-match a raw extracted test name against the local ontology.

    Returns ``None`` — leaving the row uncoded — when no alias clears
    :data:`FUZZY_MATCH_THRESHOLD`.
    """
    alias_index = _alias_index()
    if not alias_index:
        return None

    choices = list(alias_index.keys())
    best_score = -1.0
    best_key: Optional[str] = None

    for term in _candidate_terms(test_name):
        if term in alias_index:  # exact alias hit — skip fuzzy scoring entirely
            best_key, best_score = alias_index[term], 100.0
            break
        result = process.extractOne(term, choices, scorer=fuzz.WRatio)
        if result is not None:
            match_str, score, _ = result
            if score > best_score:
                best_score, best_key = score, alias_index[match_str]

    if best_key is None or best_score < FUZZY_MATCH_THRESHOLD:
        if best_key is not None:
            logger.debug(
                "Ontology match for %r rejected below threshold (%.1f < %.1f)",
                test_name, best_score, FUZZY_MATCH_THRESHOLD,
            )
        return None

    entry = _load_ontology()[best_key]
    return {
        "ontology_key": entry.key,
        "loinc_code": entry.loinc_code,
        "standard_name": entry.standard_name,
        "canonical_unit": entry.unit,
        "panel": entry.panel,
        "kind": entry.kind,
        "reference": entry.reference,
        "normal_values": entry.normal_values,
        "explanation": entry.explanation,
        "match_confidence": round(best_score, 1),
    }


# ───────────────────────── Reference range parsing ───────────────────────────


_TWO_SIDED = re.compile(r"^\s*([\d.]+)\s*[-–—]\s*([\d.]+)\s*$")
_UPPER_BOUND = re.compile(r"^\s*[<≤]=?\s*([\d.]+)\s*$")
_LOWER_BOUND = re.compile(r"^\s*[>≥]=?\s*([\d.]+)\s*$")


@dataclass
class ParsedRange:
    low: Optional[float] = None
    high: Optional[float] = None
    source: str = "report"  # "report" | "ontology_default"


def _parse_printed_range(raw: Optional[str]) -> Optional[ParsedRange]:
    """Parse a printed reference interval. Returns None for qualitative/unparsable text."""
    if not raw:
        return None
    raw = raw.strip()

    m = _TWO_SIDED.match(raw)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return ParsedRange(low=min(low, high), high=max(low, high))

    m = _UPPER_BOUND.match(raw)
    if m:
        return ParsedRange(low=None, high=float(m.group(1)))

    m = _LOWER_BOUND.match(raw)
    if m:
        return ParsedRange(low=float(m.group(1)), high=None)

    return None


def _ontology_default_range(reference: Dict[str, Any], sex: Optional[str]) -> Optional[ParsedRange]:
    """The ontology's own interval, preferring a sex-specific bucket over ``default``."""
    if not reference:
        return None
    bucket = reference.get(sex.strip().lower()) if sex else None
    bucket = bucket or reference.get("default")
    if not bucket:
        return None
    low, high = bucket.get("low"), bucket.get("high")
    if low is None and high is None:
        return None
    return ParsedRange(low=low, high=high, source="ontology_default")


_LEADING_QUALIFIER = re.compile(r"^[<>≤≥]=?\s*")


def _to_number(value: Any) -> Optional[float]:
    """Best-effort numeric parse; returns None for genuinely qualitative text ('Not seen')."""
    if value is None:
        return None
    text = _LEADING_QUALIFIER.sub("", str(value).strip()).replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


# ─────────────────────────────── Flagging ────────────────────────────────────


def evaluate_flag(
    observed_value: Any,
    printed_range: Optional[str],
    ontology_match: Optional[Dict[str, Any]],
    sex: Optional[str],
) -> Dict[str, Any]:
    """
    Decide GREEN / AMBER / RED / UNKNOWN for one row.

    Returns a dict with ``flag``, ``range_low``, ``range_high`` and
    ``range_source`` (``"report"``, ``"ontology_default"``, ``"qualitative"``
    or ``"none"``) so callers can show where the interval came from.
    """
    range_ = _parse_printed_range(printed_range)
    if range_ is None and ontology_match:
        range_ = _ontology_default_range(ontology_match.get("reference", {}), sex)

    value = _to_number(observed_value)

    # No usable numeric interval — try the qualitative path, else UNKNOWN.
    if value is None or range_ is None:
        if ontology_match and ontology_match.get("kind") == "qualitative":
            normal_values = ontology_match.get("normal_values") or []
            text = str(observed_value or "").strip().lower()
            if normal_values and text:
                flag = "GREEN" if text in normal_values else "RED"
                return {"flag": flag, "range_low": None, "range_high": None, "range_source": "qualitative"}
        return {"flag": "UNKNOWN", "range_low": None, "range_high": None, "range_source": "none"}

    low, high = range_.low, range_.high
    if low is not None and high is not None:
        margin = (high - low) * AMBER_BAND_PCT
    else:
        bound = high if high is not None else low
        margin = abs(bound) * AMBER_BAND_PCT if bound else 0.0

    if low is not None and value < low:
        flag = "AMBER" if value >= low - margin else "RED"
    elif high is not None and value > high:
        flag = "AMBER" if value <= high + margin else "RED"
    else:
        flag = "GREEN"

    return {"flag": flag, "range_low": low, "range_high": high, "range_source": range_.source}


def _format_range(range_low: Optional[float], range_high: Optional[float]) -> Optional[str]:
    """Render a fallback reference-range string when the report printed none."""
    if range_low is not None and range_high is not None:
        low = int(range_low) if float(range_low).is_integer() else range_low
        high = int(range_high) if float(range_high).is_integer() else range_high
        return f"{low} - {high}"
    if range_high is not None:
        return f"< {range_high}"
    if range_low is not None:
        return f"> {range_low}"
    return None


# ────────────────────────────── Public API ───────────────────────────────────


def process_item(item: Dict[str, Any], sex: Optional[str] = None) -> Dict[str, Any]:
    """
    Turn one Agent-1 extracted row into a coded, flagged record shaped like
    :class:`src.schemas.report.LabResult`.
    """
    test_name = item.get("test_name") or ""
    observed_value = item.get("observed_value")
    printed_range = item.get("reference_range")
    printed_unit = item.get("unit")

    match = match_to_loinc(test_name)
    verdict = evaluate_flag(observed_value, printed_range, match, sex)

    return {
        "test_name": test_name,
        "standard_name": (match or {}).get("standard_name"),
        "loinc_code": (match or {}).get("loinc_code"),
        "observed_value": observed_value,
        "unit": printed_unit or (match or {}).get("canonical_unit"),
        "reference_range": printed_range or _format_range(verdict["range_low"], verdict["range_high"]),
        "flag": verdict["flag"],
        "explanation": (match or {}).get("explanation") if match else None,
        "panel": item.get("panel") or (match or {}).get("panel"),
        "match_confidence": (match or {}).get("match_confidence"),
        "range_source": verdict["range_source"],
    }


def annotate_items(
    items: List[Dict[str, Any]], patient_info: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Run Agent 2 over every row Agent 1 extracted for one report."""
    sex = (patient_info or {}).get("sex")
    return [process_item(item, sex) for item in items]
