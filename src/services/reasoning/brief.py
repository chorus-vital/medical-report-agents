"""
The deterministic fact set Agent 3 reasons over.

The LLM never sees Agent 2's rows — it sees a rendered ``ClinicalBrief``.
That indirection is what makes verification possible: the brief *defines*
the set of numbers and analytes a claim is allowed to mention, so the
verifier has something concrete to check against rather than a vibe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_FLAGS = ("GREEN", "AMBER", "RED", "UNKNOWN")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalise_number(text: Any) -> str:
    """
    Canonical string form of a number, so "5.0", "5" and "5.00" compare equal.

    Non-numeric input comes back stripped and lowercased, which keeps
    qualitative values ("Not seen") comparable without a separate path.
    """
    if text is None:
        return ""
    raw = str(text).strip().replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return raw.lower()
    if value.is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def _as_text(value: Any) -> str:
    """String form that preserves a genuine zero, which ``or ""`` discards."""
    return "" if value is None else str(value)


@dataclass(frozen=True)
class Abnormality:
    """One row outside its reference interval."""

    test_name: str
    value: str
    unit: Optional[str]
    range_text: str
    flag: str            # "AMBER" | "RED"
    direction: str       # "below" | "above" | "outside"
    panel: Optional[str]
    loinc_code: Optional[str]
    # Agent 2's one-line gloss of what the analyte measures, in ordinary words.
    plain_meaning: Optional[str] = None


@dataclass(frozen=True)
class ClinicalBrief:
    """Everything Agent 3 is allowed to say something about."""

    total_rows: int
    coded_rows: int
    unknown_rows: int
    flag_counts: Dict[str, int]
    abnormalities: Tuple[Abnormality, ...]
    normal_panels: Tuple[str, ...]
    lab_notes: Tuple[str, ...]
    patient_age: Optional[str]
    patient_sex: Optional[str]
    extraction_degraded: bool
    allowed_numbers: frozenset
    allowed_analytes: frozenset
    allowed_loinc: frozenset
    unknown_analytes: frozenset
    # Ontology-authored glosses. Vetted repo content, so the verifier exempts
    # these exact spans from the diagnosis blocklist the way it exempts a
    # quoted lab note — "infection-fighting white blood cells" is a definition,
    # not a claim that the patient has an infection.
    allowed_glosses: frozenset


def _direction(value: Any, range_text: Optional[str]) -> str:
    """Which side of the printed interval the value falls on."""
    numbers = _NUMBER.findall(str(range_text or ""))
    observed = _NUMBER.findall(str(value or ""))
    if not numbers or not observed:
        return "outside"
    try:
        obs = float(observed[0])
    except ValueError:
        return "outside"
    bounds = [float(n) for n in numbers]
    if len(bounds) >= 2:
        if obs < min(bounds):
            return "below"
        if obs > max(bounds):
            return "above"
        return "outside"
    return "below" if obs < bounds[0] else "above"


def build_brief(
    lab_results: Sequence[Dict[str, Any]],
    patient_info: Optional[Dict[str, Any]],
    report_notes: Optional[Sequence[str]],
    extraction_degraded: bool,
) -> ClinicalBrief:
    """Freeze Agent 2's rows into the fact set the rest of Agent 3 works from."""
    rows = list(lab_results or [])
    flag_counts = {f: 0 for f in _FLAGS}
    abnormalities: List[Abnormality] = []
    allowed_numbers: set = set()
    allowed_analytes: set = set()
    allowed_loinc: set = set()
    unknown_analytes: set = set()
    evaluated_analytes: set = set()
    panels_seen: set = set()
    panels_with_abnormality: set = set()

    for row in rows:
        flag = row.get("flag") or "UNKNOWN"
        flag_counts[flag] = flag_counts.get(flag, 0) + 1

        name = row.get("standard_name") or row.get("test_name") or ""
        names = {n.strip().lower() for n in (name, row.get("test_name") or "") if n}
        allowed_analytes.update(names)
        if flag == "UNKNOWN":
            unknown_analytes.update(names)
        else:
            evaluated_analytes.update(names)

        if row.get("loinc_code"):
            allowed_loinc.add(row["loinc_code"])

        allowed_numbers.add(normalise_number(row.get("observed_value")))
        for bound in _NUMBER.findall(str(row.get("reference_range") or "")):
            allowed_numbers.add(normalise_number(bound))

        panel = row.get("panel")
        if panel:
            panels_seen.add(panel)

        if flag in ("AMBER", "RED"):
            abnormalities.append(
                Abnormality(
                    test_name=name or row.get("test_name") or "Unnamed test",
                    value=_as_text(row.get("observed_value")),
                    unit=row.get("unit"),
                    range_text=str(row.get("reference_range") or ""),
                    flag=flag,
                    direction=_direction(row.get("observed_value"),
                                         row.get("reference_range")),
                    panel=panel,
                    loinc_code=row.get("loinc_code"),
                    plain_meaning=row.get("explanation") or None,
                )
            )
            if panel:
                panels_with_abnormality.add(panel)

    abnormalities.sort(key=lambda a: (0 if a.flag == "RED" else 1, a.test_name))

    coded_rows = sum(1 for r in rows if r.get("loinc_code"))
    counts = (len(rows), coded_rows, flag_counts["UNKNOWN"],
              flag_counts["RED"], flag_counts["AMBER"], flag_counts["GREEN"],
              len(abnormalities))
    allowed_numbers.update(normalise_number(c) for c in counts)

    age = (patient_info or {}).get("age")
    if age:
        for token in _NUMBER.findall(str(age)):
            allowed_numbers.add(normalise_number(token))

    return ClinicalBrief(
        total_rows=len(rows),
        coded_rows=coded_rows,
        unknown_rows=flag_counts["UNKNOWN"],
        flag_counts=flag_counts,
        abnormalities=tuple(abnormalities),
        normal_panels=tuple(sorted(panels_seen - panels_with_abnormality)),
        lab_notes=tuple(n for n in (report_notes or []) if n and n.strip()),
        patient_age=age,
        patient_sex=(patient_info or {}).get("sex"),
        extraction_degraded=bool(extraction_degraded),
        allowed_numbers=frozenset(allowed_numbers),
        allowed_analytes=frozenset(allowed_analytes),
        allowed_loinc=frozenset(allowed_loinc),
        # A name evaluated on any row is evaluated: labs routinely print the
        # same analyte twice (fasting and random glucose), and one UNKNOWN row
        # must not make the other row's result unspeakable.
        unknown_analytes=frozenset(unknown_analytes - evaluated_analytes),
        allowed_glosses=frozenset(
            a.plain_meaning for a in abnormalities if a.plain_meaning
        ),
    )


def render_brief(brief: ClinicalBrief) -> str:
    """The brief as compact text, for the LLM prompt."""
    lines: List[str] = [
        f"Total results: {brief.total_rows}",
        f"Normal: {brief.flag_counts['GREEN']}   "
        f"Borderline: {brief.flag_counts['AMBER']}   "
        f"Outside range: {brief.flag_counts['RED']}   "
        f"Not evaluated: {brief.flag_counts['UNKNOWN']}",
    ]
    if brief.patient_age or brief.patient_sex:
        lines.append(f"Patient: {brief.patient_age or '?'} / {brief.patient_sex or '?'}")

    if brief.abnormalities:
        lines.append("")
        lines.append("RESULTS OUTSIDE THEIR REFERENCE INTERVAL:")
        for a in brief.abnormalities:
            unit = f" {a.unit}" if a.unit else ""
            lines.append(
                f"  - {a.test_name}: {a.value}{unit} ({a.flag}, {a.direction} "
                f"the interval {a.range_text})"
            )
            if a.plain_meaning:
                lines.append(f"      what it measures: {a.plain_meaning}")
    else:
        lines.append("")
        lines.append("No result is outside its reference interval.")

    if brief.unknown_analytes:
        lines.append("")
        lines.append(
            "NOT EVALUATED — say nothing about whether these are normal: "
            + ", ".join(sorted(brief.unknown_analytes))
        )

    if brief.normal_panels:
        lines.append("")
        lines.append(f"Panels with nothing abnormal: {', '.join(brief.normal_panels)}")

    if brief.lab_notes:
        lines.append("")
        lines.append("NOTES PRINTED ON THE REPORT BY THE LAB (quote verbatim only):")
        for note in brief.lab_notes:
            lines.append(f'  "{note}"')

    if brief.extraction_degraded:
        lines.append("")
        lines.append("WARNING: extraction was degraded; the row list may be incomplete.")

    return "\n".join(lines)
