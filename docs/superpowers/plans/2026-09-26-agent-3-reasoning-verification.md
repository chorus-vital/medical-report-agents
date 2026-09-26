# Agent 3 — Reasoning & Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Agent 2's coded, flagged lab rows into a plain-language analysis whose every factual claim is verified in code against those rows.

**Architecture:** A deterministic brief is built from the rows and is the only thing the LLM sees. The LLM writes narrative; a non-LLM verifier checks every number and analyte in that narrative against the brief and drops or regenerates what it cannot trace. Four of the five modules never touch the network, so the whole layer is testable with no API key.

**Tech Stack:** Python 3.11+, Pydantic v2, `google-genai` (Gemini, structured output via `response_schema`), LangGraph, pytest, rapidfuzz (already a dependency, reused by the verifier's analyte scan).

**Spec:** `docs/superpowers/specs/2026-09-26-agent-3-reasoning-verification-design.md`

## Global Constraints

- Agent 3 never names a candidate diagnosis or condition. Facts and urgency only.
- `confidence_score` is a float in `[0.0, 1.0]`, computed in code, never supplied by the LLM.
- The LLM sees the rendered `ClinicalBrief` and nothing else — never raw `lab_results`.
- Every module except `narrative.py`'s LLM path must be importable and testable with no `GEMINI_API_KEY`.
- The node never raises. Failures append to `errors` and the pipeline completes.
- Tests that hit the real API carry `@pytest.mark.live`; `pytest.ini` already deselects that marker by default.
- Follow the existing retry helper `_with_retries` and client factory `get_vision_model()` rather than writing new ones.
- Commit messages: no `Co-Authored-By` and no "Generated with Claude Code" lines.

### Correction to the spec

The spec's escalation section says rules match on `ontology_key`. **`process_item()` does not emit `ontology_key`** — it emits `loinc_code`. Rules therefore match on `loinc_code`, which is the stable clinical identifier anyway. Codes verified against `data/lab_ontology.json`:

| Analyte | LOINC |
|---|---|
| Absolute Neutrophil Count | `751-8` |
| Platelet Count | `777-3` |
| Hemoglobin | `718-7` |
| Potassium | `2823-3` |
| Fasting Blood Glucose | `1558-6` |
| Post-prandial Glucose | `1521-4` |

## Review Focus

Five conditions the spec implies but does not give a task. Each has a test assigned to the task that owns the code.

1. **A verbatim lab note that itself contains condition vocabulary.** The test report's note reads *"Advice to rule out viral etiology in view of leukopenia."* A naive diagnosis-language blocklist rejects the lab's own words. The verifier must exempt verbatim note spans. → Task 5, Step 9.
2. **Zero lab rows.** An unreadable PDF yields `lab_results == []`; the confidence formula divides by `total_rows`. → Task 8, Step 7.
3. **A row whose `observed_value` is `None` or non-numeric.** Agent 2 emits these with `flag == "UNKNOWN"`; the brief formats every row into strings. → Task 3, Step 7.
4. **An LLM reply that is schema-valid but empty** (`summary == ""`, all lists empty). Structured output guarantees shape, not content. → Task 7, Step 7.
5. **The same analyte appearing twice** (many labs print a test in two panels). Counts and `allowed_analytes` must not be corrupted. → Task 3, Step 9.

---

## File Structure

| Path | Responsibility | Status |
|---|---|---|
| `src/services/terminology.py` | AMBER band fix in `evaluate_flag` | modify |
| `src/services/reasoning/__init__.py` | `analyze()`, `score_confidence()` | create |
| `src/services/reasoning/brief.py` | `Abnormality`, `ClinicalBrief`, `build_brief()` | create |
| `src/services/reasoning/escalation.py` | `FiredRule`, `Escalation`, `escalate()` | create |
| `src/services/reasoning/verify.py` | `Violation`, `VerificationReport`, `verify()` | create |
| `src/services/reasoning/narrative.py` | `NarrativeDraft`, `render_fallback()`, `narrate_llm()` | create |
| `src/graph/nodes.py` | implement `reason_and_verify_node` | modify |
| `src/schemas/state.py` | 3 new state fields | modify |
| `src/schemas/report.py` | 3 new `ReportAnalysis` fields | modify |
| `src/api/routes/reports.py` | `reasoning` block in `_build_response` | modify |
| `src/static/index.html` | render analysis + escalation banner | modify |
| `tests/fixtures/dev_chavan_agent2.json` | real-report fixture | create |
| `tests/test_reasoning_brief.py` | | create |
| `tests/test_reasoning_escalation.py` | | create |
| `tests/test_reasoning_verify.py` | | create |
| `tests/test_reasoning_narrative.py` | | create |
| `tests/test_reasoning_analyze.py` | | create |
| `tests/test_pipeline_real_report.py` | integration, offline + live | create |

---

## Task 1: Fix Agent 2's AMBER band

The current margin is 10% of the interval *width*, so range `20 - 500` tolerates a value of `0` as "borderline". Escalation derives from these flags, so this lands first. The new rule — 10% of the breached bound's own magnitude — is what `evaluate_flag` already does for one-sided ranges, so this makes two-sided consistent with one-sided.

**Files:**
- Modify: `src/services/terminology.py` (the `AMBER_BAND_PCT` comment, and the margin block in `evaluate_flag`)
- Test: `tests/test_terminology.py`

**Interfaces:**
- Consumes: nothing
- Produces: `evaluate_flag(observed_value, printed_range, ontology_match, sex) -> dict` — unchanged signature, changed AMBER/RED boundary

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_terminology.py`:

```python
# The AMBER band is a fraction of the breached bound's own magnitude, not of
# the interval width. With a width-based band, range "20 - 500" gave a margin
# of 48, so even a value of 0 came back AMBER — "borderline" for a result that
# is 100% below the lower limit.
@pytest.mark.parametrize(
    "value,range_text,expected",
    [
        ("0", "20 - 500", "RED"),      # was AMBER under the width rule
        ("13", "20 - 500", "RED"),     # real AEC row from a CBC report
        ("19", "20 - 500", "AMBER"),   # genuinely marginal
        ("81.1", "83 - 101", "AMBER"), # real MCV row; was RED under the width rule
        ("74", "83 - 101", "RED"),
    ],
)
def test_amber_band_scales_with_the_breached_bound(value, range_text, expected):
    assert term.evaluate_flag(value, range_text, None, None)["flag"] == expected


def test_amber_band_is_symmetric_on_the_upper_bound():
    assert term.evaluate_flag("209", "20 - 200", None, None)["flag"] == "AMBER"
    assert term.evaluate_flag("260", "20 - 200", None, None)["flag"] == "RED"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_terminology.py -q -k amber_band`
Expected: FAIL — `"0"` and `"13"` return `AMBER`, `"81.1"` returns `RED`.

- [ ] **Step 3: Replace the margin block**

In `src/services/terminology.py`, find this block in `evaluate_flag`:

```python
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
```

Replace it with:

```python
    low, high = range_.low, range_.high
    if low is not None and value < low:
        margin = abs(low) * AMBER_BAND_PCT
        flag = "AMBER" if value >= low - margin else "RED"
    elif high is not None and value > high:
        margin = abs(high) * AMBER_BAND_PCT
        flag = "AMBER" if value <= high + margin else "RED"
    else:
        flag = "GREEN"
```

Update the `AMBER_BAND_PCT` comment to match:

```python
# How far outside a boundary still counts as "borderline" (AMBER) rather than
# "alert" (RED), as a fraction of the breached bound's own magnitude. Scaling
# by the interval *width* instead lets a wide range swallow an extreme value:
# with "20 - 500" the band was 48, so a result of 0 read as borderline.
AMBER_BAND_PCT = 0.10
```

- [ ] **Step 4: Run the new tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_terminology.py -q -k amber_band`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS. Every pre-existing flag test was chosen to survive this change — `95`/`100 - 200` → AMBER (margin 10), `205` → AMBER (margin 20), `50` and `260` → RED, `215`/`< 200` → AMBER. If any of those fail, stop: the rule is wrong, not the test.

- [ ] **Step 6: Commit**

```bash
git add src/services/terminology.py tests/test_terminology.py
git commit -m "Scale the AMBER band to the breached bound, not the interval width"
```

---

## Task 2: Capture the real-report fixture

Every later task tests against Agent 2's genuine output for `D:\medical report analyzer data\Dev chavan Report.pdf`. Capturing it once, as committed JSON, keeps those tests offline and deterministic.

**Files:**
- Create: `tests/fixtures/dev_chavan_agent2.json`
- Create: `scripts/capture_fixture.py`

**Interfaces:**
- Consumes: `src.graph.pipeline.pipeline`
- Produces: a JSON object `{"lab_results": [...], "patient_info": {...}, "report_notes": [...], "extraction_degraded": bool}`

- [ ] **Step 1: Write the capture script**

Create `scripts/capture_fixture.py`:

```python
"""
Capture Agent 1 + Agent 2 output for a real report as a test fixture.

Re-run this when the ontology or the extractor changes in a way that should
change the fixture. Committing the output keeps the reasoning tests offline.

Usage:
    python scripts/capture_fixture.py "<path to report.pdf>" tests/fixtures/<name>.json
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.pipeline import pipeline  # noqa: E402


async def main(pdf_path: str, out_path: str) -> None:
    state = await pipeline.ainvoke({"file_path": pdf_path, "file_type": "pdf"})
    fixture = {
        "source_file": Path(pdf_path).name,
        "extraction_method": state.get("extraction_method"),
        "extraction_degraded": bool(state.get("extraction_degraded")),
        "patient_info": state.get("patient_info"),
        "report_notes": state.get("report_notes") or [],
        "lab_results": state.get("lab_results") or [],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"wrote {out_path}: {len(fixture['lab_results'])} rows, "
          f"method={fixture['extraction_method']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 2: Run it against the real report**

```bash
venv/Scripts/python.exe scripts/capture_fixture.py \
  "D:\medical report analyzer data\Dev chavan Report.pdf" \
  tests/fixtures/dev_chavan_agent2.json
```

Gemini returns 503 intermittently. If `extraction_method` comes back as `regex-fallback:...`, re-run up to three times to try to capture a clean LLM extraction. If it stays degraded, keep the regex-fallback fixture — it still produced all 22 rows — and note it in the commit message.

- [ ] **Step 3: Hand-verify the fixture against the PDF**

Open the fixture and confirm these rows and flags are present. This is the check that the fixture is real data and not a capture bug. Flags are post-Task-1.

| Test | Value | Range | Flag |
|---|---|---|---|
| Total White Blood Cell Count (TC) | 2130 | 4000 - 10000 | RED |
| Absolute Neutrophil Count (ANC) | 1035 | 2000 - 7000 | RED |
| Absolute Lymphocyte Count (ALC) | 878 | 1000 - 3000 | RED |
| Absolute Monocyte Count (AMC) | 192 | 200 - 1000 | AMBER |
| Absolute Eosinophil Count (AEC) | 13 | 20 - 500 | RED |
| Mean Corpuscular Volume (MCV) | 81.1 | 83 - 101 | AMBER |
| Lymphocytes | 41.2 | 20 - 40 | AMBER |
| Hemoglobin (Hb) | 15.3 | 13 - 17 | GREEN |
| Platelet Count | 194 | 150 - 450 | GREEN |

Confirm `Total White Blood Cell Count (TC)` now carries `loinc_code: "6690-2"` — it was uncoded before the alias fix in `26afaba`.

- [ ] **Step 4: Commit**

```bash
git add scripts/capture_fixture.py tests/fixtures/dev_chavan_agent2.json
git commit -m "Add real-report Agent 2 fixture and the script that captures it"
```

---

## Task 3: The clinical brief

**Files:**
- Create: `src/services/reasoning/__init__.py` (empty for now), `src/services/reasoning/brief.py`
- Test: `tests/test_reasoning_brief.py`

**Interfaces:**
- Consumes: Agent 2 row dicts as produced by `terminology.process_item`
- Produces:
  - `Abnormality(test_name: str, value: str, unit: str|None, range_text: str, flag: str, direction: str, panel: str|None, loinc_code: str|None)`
  - `ClinicalBrief` with fields `total_rows, coded_rows, unknown_rows, flag_counts, abnormalities, normal_panels, lab_notes, patient_age, patient_sex, extraction_degraded, allowed_numbers, allowed_analytes`
  - `build_brief(lab_results, patient_info, report_notes, extraction_degraded) -> ClinicalBrief`
  - `render_brief(brief) -> str`
  - `normalise_number(text) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reasoning_brief.py`:

```python
"""Tests for the deterministic clinical brief (Agent 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.reasoning import brief as b

FIXTURE = Path(__file__).parent / "fixtures" / "dev_chavan_agent2.json"


@pytest.fixture(scope="module")
def dev_chavan():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_brief(dev_chavan):
    return b.build_brief(
        dev_chavan["lab_results"],
        dev_chavan["patient_info"],
        dev_chavan["report_notes"],
        dev_chavan["extraction_degraded"],
    )


def test_brief_counts_every_row(dev_brief):
    assert dev_brief.total_rows == 22
    assert sum(dev_brief.flag_counts.values()) == dev_brief.total_rows


def test_red_abnormalities_come_before_amber(dev_brief):
    flags = [a.flag for a in dev_brief.abnormalities]
    assert flags == sorted(flags, key=lambda f: 0 if f == "RED" else 1)
    assert "GREEN" not in flags


def test_abnormality_records_direction(dev_brief):
    wbc = next(a for a in dev_brief.abnormalities if "White Blood Cell" in a.test_name)
    assert wbc.value == "2130"
    assert wbc.direction == "below"
    assert wbc.flag == "RED"


def test_allowed_numbers_hold_values_and_bounds(dev_brief):
    assert "2130" in dev_brief.allowed_numbers    # an observed value
    assert "4000" in dev_brief.allowed_numbers    # a range bound
    assert "22" in dev_brief.allowed_numbers      # a count
    assert "9999" not in dev_brief.allowed_numbers


def test_allowed_analytes_hold_only_tested_rows(dev_brief):
    assert "hemoglobin" in dev_brief.allowed_analytes
    assert "creatinine" not in dev_brief.allowed_analytes  # never tested here


def test_lab_notes_are_carried_verbatim():
    brief = b.build_brief([], None, ["Advice to rule out viral etiology."], False)
    assert brief.lab_notes == ("Advice to rule out viral etiology.",)


def test_non_numeric_value_does_not_crash_the_brief():
    # Agent 2 emits UNKNOWN rows with a None or qualitative value.
    rows = [
        {"test_name": "Malarial Parasite", "observed_value": None, "unit": None,
         "reference_range": None, "flag": "UNKNOWN", "panel": None, "loinc_code": None},
        {"test_name": "Colour", "observed_value": "Pale yellow", "unit": None,
         "reference_range": "Pale yellow", "flag": "UNKNOWN", "panel": None,
         "loinc_code": None},
    ]
    brief = b.build_brief(rows, None, [], False)
    assert brief.total_rows == 2
    assert brief.unknown_rows == 2
    assert brief.abnormalities == ()


def test_duplicate_analyte_rows_are_counted_once_in_allowed_analytes():
    rows = [
        {"test_name": "Hemoglobin", "standard_name": "Hemoglobin", "observed_value": "15.3",
         "unit": "g/dL", "reference_range": "13 - 17", "flag": "GREEN",
         "panel": "CBC", "loinc_code": "718-7"},
        {"test_name": "Hemoglobin", "standard_name": "Hemoglobin", "observed_value": "15.3",
         "unit": "g/dL", "reference_range": "13 - 17", "flag": "GREEN",
         "panel": "CBC", "loinc_code": "718-7"},
    ]
    brief = b.build_brief(rows, None, [], False)
    assert brief.total_rows == 2
    assert brief.allowed_analytes == frozenset({"hemoglobin"})


@pytest.mark.parametrize(
    "raw,expected",
    [("5.0", "5"), ("5", "5"), ("0.171", "0.171"), ("1,200", "1200"), ("13.20", "13.2")],
)
def test_normalise_number(raw, expected):
    assert b.normalise_number(raw) == expected


def test_render_brief_mentions_every_abnormality(dev_brief):
    text = b.render_brief(dev_brief)
    for a in dev_brief.abnormalities:
        assert a.test_name in text
        assert a.value in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_brief.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.reasoning'`

- [ ] **Step 3: Create the package**

```bash
mkdir -p src/services/reasoning
```

Create `src/services/reasoning/__init__.py` containing only:

```python
"""Agent 3 — Reasoning & Verification."""
```

- [ ] **Step 4: Write `brief.py`**

Create `src/services/reasoning/brief.py`:

```python
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
    raw = str(text or "").strip().replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return raw.lower()
    if value.is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


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
    panels_seen: set = set()
    panels_with_abnormality: set = set()

    for row in rows:
        flag = row.get("flag") or "UNKNOWN"
        flag_counts[flag] = flag_counts.get(flag, 0) + 1

        name = row.get("standard_name") or row.get("test_name") or ""
        if name:
            allowed_analytes.add(name.strip().lower())
        printed = (row.get("test_name") or "").strip().lower()
        if printed:
            allowed_analytes.add(printed)

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
                    value=str(row.get("observed_value") or ""),
                    unit=row.get("unit"),
                    range_text=str(row.get("reference_range") or ""),
                    flag=flag,
                    direction=_direction(row.get("observed_value"),
                                         row.get("reference_range")),
                    panel=panel,
                    loinc_code=row.get("loinc_code"),
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
    else:
        lines.append("")
        lines.append("No result is outside its reference interval.")

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
```

- [ ] **Step 5: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_brief.py -q`
Expected: PASS

- [ ] **Step 6: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 7: Confirm Review Focus item 3 is covered**

`test_non_numeric_value_does_not_crash_the_brief` is in Step 1 and must pass. If it was removed, restore it.

- [ ] **Step 8: Commit**

```bash
git add src/services/reasoning/__init__.py src/services/reasoning/brief.py tests/test_reasoning_brief.py
git commit -m "Add Agent 3's deterministic clinical brief"
```

- [ ] **Step 9: Confirm Review Focus item 5 is covered**

`test_duplicate_analyte_rows_are_counted_once_in_allowed_analytes` is in Step 1 and must pass.

---

## Task 4: Escalation rules

**Files:**
- Create: `src/services/reasoning/escalation.py`
- Test: `tests/test_reasoning_escalation.py`

**Interfaces:**
- Consumes: `ClinicalBrief` from Task 3
- Produces:
  - `FiredRule(rule_id: str, test_name: str, detail: str)`
  - `Escalation(level: str, reasons: tuple[FiredRule, ...])`
  - `escalate(brief) -> Escalation`
  - `LEVELS: tuple[str, ...]` ordered least to most urgent
  - `LEVEL_TEXT: dict[str, str]` patient-facing wording

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reasoning_escalation.py`:

```python
"""Tests for Agent 3's deterministic escalation rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.reasoning import brief as b
from src.services.reasoning import escalation as e

FIXTURE = Path(__file__).parent / "fixtures" / "dev_chavan_agent2.json"


def _brief(rows, notes=None, degraded=False):
    return b.build_brief(rows, None, notes or [], degraded)


def _row(name, value, rng, flag, loinc=None):
    return {"test_name": name, "standard_name": name, "observed_value": value,
            "unit": None, "reference_range": rng, "flag": flag,
            "panel": "CBC", "loinc_code": loinc}


def test_all_green_is_routine():
    assert e.escalate(_brief([_row("Hemoglobin", "15", "13 - 17", "GREEN", "718-7")])).level == "routine"


def test_amber_only_is_next_visit():
    result = e.escalate(_brief([_row("MCV", "81.1", "83 - 101", "AMBER", "787-2")]))
    assert result.level == "discuss_at_next_visit"
    assert [r.rule_id for r in result.reasons] == ["any_amber"]


def test_any_red_is_promptly():
    result = e.escalate(_brief([_row("Eosinophils", "13", "20 - 500", "RED", "711-2")]))
    assert result.level == "see_doctor_promptly"


@pytest.mark.parametrize(
    "name,value,rng,loinc,rule_id",
    [
        ("Absolute Neutrophil Count", "400", "2000 - 7000", "751-8", "anc_critical"),
        ("Platelet Count", "15000", "150000 - 450000", "777-3", "platelet_critical"),
        ("Hemoglobin", "6.2", "13 - 17", "718-7", "hb_critical"),
        ("Potassium", "2.1", "3.5 - 5.1", "2823-3", "potassium_critical"),
        ("Potassium", "7.0", "3.5 - 5.1", "2823-3", "potassium_critical"),
        ("Fasting Blood Glucose", "520", "70 - 100", "1558-6", "glucose_critical"),
    ],
)
def test_critical_values_escalate_to_seek_care_now(name, value, rng, loinc, rule_id):
    result = e.escalate(_brief([_row(name, value, rng, "RED", loinc)]))
    assert result.level == "seek_care_now"
    assert rule_id in [r.rule_id for r in result.reasons]


def test_moderate_neutropenia_is_promptly_not_urgent():
    result = e.escalate(_brief([_row("Absolute Neutrophil Count", "1035",
                                     "2000 - 7000", "751-8")]))
    assert result.level == "see_doctor_promptly"
    assert "anc_moderate" in [r.rule_id for r in result.reasons]


def test_highest_level_wins():
    rows = [
        _row("MCV", "81.1", "83 - 101", "AMBER", "787-2"),
        _row("Absolute Neutrophil Count", "400", "2000 - 7000", "751-8"),
    ]
    assert e.escalate(_brief(rows)).level == "seek_care_now"


def test_uncoded_row_cannot_fire_a_threshold_rule():
    # No loinc_code: Agent 2 was not confident, so the rule must not fire.
    result = e.escalate(_brief([_row("Neutrophils something", "400",
                                     "2000 - 7000", "RED", None)]))
    assert result.level == "see_doctor_promptly"          # any_red still fires
    assert "anc_critical" not in [r.rule_id for r in result.reasons]


def test_aggregate_rule_cites_the_most_severe_row():
    rows = [
        _row("MCV", "81.1", "83 - 101", "AMBER", "787-2"),
        _row("Eosinophils", "13", "20 - 500", "RED", "711-2"),
    ]
    reason = next(r for r in e.escalate(_brief(rows)).reasons if r.rule_id == "any_red")
    assert reason.test_name == "Eosinophils"


def test_every_level_has_patient_facing_text():
    for level in e.LEVELS:
        assert e.LEVEL_TEXT[level]


def test_real_report_escalates_promptly():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    brief = b.build_brief(data["lab_results"], data["patient_info"],
                          data["report_notes"], data["extraction_degraded"])
    result = e.escalate(brief)
    assert result.level == "see_doctor_promptly"
    assert "anc_moderate" in [r.rule_id for r in result.reasons]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_escalation.py -q`
Expected: FAIL — `cannot import name 'escalation'`

- [ ] **Step 3: Write `escalation.py`**

Create `src/services/reasoning/escalation.py`:

```python
"""
How soon a result should be acted on.

This is the only clinical content in the codebase. Every threshold below is a
published critical value, every rule cites the row that fired it, and none of
them names a cause — they answer "how soon", never "why". Rules key on
``loinc_code`` so they can only fire on a row Agent 2 coded confidently; an
uncoded row lowers confidence instead of triggering an alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.services.reasoning.brief import ClinicalBrief

logger = logging.getLogger(__name__)

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


# Published critical and action thresholds. Units follow what the ontology
# stores for each analyte.
_THRESHOLDS: Tuple[_Threshold, ...] = (
    _Threshold("anc_critical", "751-8", "seek_care_now",
               lambda v: v < 500, "below 500 /mm3"),
    _Threshold("platelet_critical", "777-3", "seek_care_now",
               lambda v: v < 20_000, "below 20,000 /mm3"),
    _Threshold("hb_critical", "718-7", "seek_care_now",
               lambda v: v < 7, "below 7 g/dL"),
    _Threshold("potassium_critical", "2823-3", "seek_care_now",
               lambda v: v < 2.5 or v > 6.5, "outside 2.5-6.5 mmol/L"),
    _Threshold("glucose_critical", "1558-6", "seek_care_now",
               lambda v: v < 50 or v > 500, "outside 50-500 mg/dL"),
    _Threshold("glucose_critical", "1521-4", "seek_care_now",
               lambda v: v < 50 or v > 500, "outside 50-500 mg/dL"),
    _Threshold("anc_moderate", "751-8", "see_doctor_promptly",
               lambda v: v < 1500, "below 1500 /mm3"),
    _Threshold("platelet_low", "777-3", "see_doctor_promptly",
               lambda v: v < 50_000, "below 50,000 /mm3"),
    _Threshold("hb_low", "718-7", "see_doctor_promptly",
               lambda v: v < 10, "below 10 g/dL"),
)


def _as_float(text: str) -> Optional[float]:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


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
                fired = rule.test(value)
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
        rule_level = next((r.level for r in _THRESHOLDS if r.rule_id == reason.rule_id),
                          None)
        if reason.rule_id == "any_red":
            rule_level = "see_doctor_promptly"
        elif reason.rule_id == "any_amber":
            rule_level = "discuss_at_next_visit"
        if rule_level and LEVELS.index(rule_level) > LEVELS.index(level):
            level = rule_level

    return Escalation(level=level, reasons=tuple(reasons))
```

- [ ] **Step 4: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_escalation.py -q`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/services/reasoning/escalation.py tests/test_reasoning_escalation.py
git commit -m "Add Agent 3's escalation rule pack"
```

---

## Task 5: The verifier

**Files:**
- Create: `src/services/reasoning/verify.py`
- Test: `tests/test_reasoning_verify.py`

**Interfaces:**
- Consumes: `ClinicalBrief` from Task 3; a draft object exposing `summary: str`, `key_findings: list[str]`, `doctor_questions: list[str]`, `lifestyle_tips: list[str]`
- Produces:
  - `Violation(kind: str, field: str, text: str, detail: str)`
  - `VerificationReport(clean: bool, summary_ok: bool, violations: tuple[Violation, ...], kept: dict[str, list[str]])`
  - `verify(draft, brief) -> VerificationReport`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reasoning_verify.py`:

```python
"""Tests for Agent 3's claim verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from src.services.reasoning import brief as b
from src.services.reasoning import verify as v


@dataclass
class Draft:
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    doctor_questions: List[str] = field(default_factory=list)
    lifestyle_tips: List[str] = field(default_factory=list)


@pytest.fixture
def brief():
    rows = [
        {"test_name": "Hemoglobin", "standard_name": "Hemoglobin",
         "observed_value": "15.3", "unit": "g/dL", "reference_range": "13 - 17",
         "flag": "GREEN", "panel": "CBC", "loinc_code": "718-7"},
        {"test_name": "Total White Blood Cell Count (TC)",
         "standard_name": "Total WBC Count", "observed_value": "2130",
         "unit": "cells/mm3", "reference_range": "4000 - 10000", "flag": "RED",
         "panel": "CBC", "loinc_code": "6690-2"},
        {"test_name": "Malarial Parasite", "standard_name": None,
         "observed_value": "Not seen", "unit": None, "reference_range": None,
         "flag": "UNKNOWN", "panel": None, "loinc_code": None},
    ]
    return b.build_brief(rows, None,
                         ["Advice to rule out viral etiology in view of leukopenia."],
                         False)


def test_clean_draft_passes(brief):
    draft = Draft(
        summary="Your white blood cell count is 2130, below the 4000 to 10000 range.",
        key_findings=["Total WBC Count 2130 is below the reference interval"],
    )
    report = v.verify(draft, brief)
    assert report.clean
    assert report.violations == ()


def test_fabricated_number_is_caught(brief):
    draft = Draft(summary="Your white blood cell count is 3150.")
    report = v.verify(draft, brief)
    assert not report.clean
    assert any(x.kind == "number" and "3150" in x.detail for x in report.violations)


def test_untested_analyte_is_caught(brief):
    draft = Draft(key_findings=["Your creatinine is within the normal range."])
    report = v.verify(draft, brief)
    assert any(x.kind == "analyte" for x in report.violations)
    assert report.kept["key_findings"] == []


def test_diagnosis_language_is_caught(brief):
    draft = Draft(summary="This indicates anaemia and you have an infection.")
    report = v.verify(draft, brief)
    assert any(x.kind == "diagnosis" for x in report.violations)
    assert not report.summary_ok


def test_claim_about_an_unknown_row_is_caught(brief):
    draft = Draft(key_findings=["Your Malarial Parasite result is normal."])
    report = v.verify(draft, brief)
    assert any(x.kind == "unknown_row" for x in report.violations)


def test_only_the_offending_bullet_is_dropped(brief):
    draft = Draft(
        summary="Your white blood cell count is 2130.",
        key_findings=[
            "Total WBC Count 2130 is below the reference interval",
            "Your creatinine is fine.",
        ],
    )
    report = v.verify(draft, brief)
    assert report.kept["key_findings"] == [
        "Total WBC Count 2130 is below the reference interval"
    ]
    assert report.summary_ok


def test_verbatim_lab_note_is_not_treated_as_diagnosis_language(brief):
    # The lab's own note contains "rule out viral etiology ... leukopenia".
    # Quoting it verbatim must be allowed; the blocklist applies to our prose.
    note = brief.lab_notes[0]
    draft = Draft(summary=f'Your lab added this note: "{note}"')
    report = v.verify(draft, brief)
    assert report.summary_ok, [x.detail for x in report.violations]
    assert report.clean


def test_paraphrasing_a_lab_note_is_still_caught(brief):
    draft = Draft(summary="Your lab thinks you have a viral infection.")
    report = v.verify(draft, brief)
    assert not report.summary_ok


def test_counts_are_quotable(brief):
    draft = Draft(summary="1 of 3 results is outside its reference interval.")
    assert v.verify(draft, brief).clean
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_verify.py -q`
Expected: FAIL — `cannot import name 'verify'`

- [ ] **Step 3: Write `verify.py`**

Create `src/services/reasoning/verify.py`:

```python
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
from typing import Any, Dict, List, Tuple

from src.services.reasoning.brief import ClinicalBrief, normalise_number
from src.services.terminology import _alias_index

_FIELDS = ("summary", "key_findings", "doctor_questions", "lifestyle_tips")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

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


def _unknown_row_names(brief: ClinicalBrief) -> frozenset:
    """Analyte names belonging to rows Agent 2 could not evaluate."""
    abnormal = {a.test_name.lower() for a in brief.abnormalities}
    return frozenset(n for n in brief.allowed_analytes if n not in abnormal)


def _check(text: str, field: str, brief: ClinicalBrief,
           unknown_names: frozenset) -> List[Violation]:
    found: List[Violation] = []
    if not text or not text.strip():
        return found

    for raw in _NUMBER.findall(text):
        token = normalise_number(raw)
        if token in _FREE_NUMBERS or token in brief.allowed_numbers:
            continue
        found.append(Violation("number", field, text,
                               f"{raw} appears in no row of this report"))

    lowered = text.lower()
    for alias, _key in _alias_index().items():
        if len(alias) < 5:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", lowered) and \
                alias not in brief.allowed_analytes:
            found.append(Violation("analyte", field, text,
                                   f"'{alias}' was not tested in this report"))
            break

    scannable = _strip_lab_notes(text, brief)
    for pattern in _DIAGNOSIS_PATTERNS:
        match = pattern.search(scannable)
        if match:
            found.append(Violation("diagnosis", field, text,
                                   f"diagnosis language: '{match.group(0)}'"))
            break

    for name in unknown_names:
        if len(name) < 5:
            continue
        if re.search(rf"\b{re.escape(name)}\b", lowered) and \
                re.search(r"\b(normal|abnormal|fine|healthy|within range)\b", lowered):
            found.append(Violation("unknown_row", field, text,
                                   f"'{name}' was not evaluated, so it cannot be "
                                   f"called normal or abnormal"))
            break

    return found


def verify(draft: Any, brief: ClinicalBrief) -> VerificationReport:
    """Check every claim in ``draft`` against ``brief``."""
    violations: List[Violation] = []
    kept: Dict[str, List[str]] = {}
    unknown_names = _unknown_row_names(brief)

    summary_violations = _check(getattr(draft, "summary", "") or "",
                                "summary", brief, unknown_names)
    violations.extend(summary_violations)

    for field in ("key_findings", "doctor_questions", "lifestyle_tips"):
        surviving: List[str] = []
        for item in getattr(draft, field, None) or []:
            item_violations = _check(item, field, brief, unknown_names)
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
```

- [ ] **Step 4: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_verify.py -q`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/services/reasoning/verify.py tests/test_reasoning_verify.py
git commit -m "Add Agent 3's claim verifier"
```

- [ ] **Step 7: Confirm Review Focus item 1 is covered**

`test_verbatim_lab_note_is_not_treated_as_diagnosis_language` and
`test_paraphrasing_a_lab_note_is_still_caught` are both in Step 1 and must pass.
Together they pin the exemption to exact spans only.

---

## Task 6: The fallback narrative renderer

Built before the LLM path because it is the guaranteed floor: it runs on every LLM failure and in every offline test.

**Files:**
- Create: `src/services/reasoning/narrative.py`
- Test: `tests/test_reasoning_narrative.py`

**Interfaces:**
- Consumes: `ClinicalBrief` (Task 3), `Escalation` (Task 4)
- Produces:
  - `NarrativeDraft(BaseModel)` with `summary: str`, `key_findings: list[str]`, `doctor_questions: list[str]`, `lifestyle_tips: list[str]`
  - `render_fallback(brief, escalation) -> NarrativeDraft`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reasoning_narrative.py`:

```python
"""Tests for Agent 3's narrative rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.reasoning import brief as b
from src.services.reasoning import escalation as e
from src.services.reasoning import narrative as n
from src.services.reasoning import verify as v

FIXTURE = Path(__file__).parent / "fixtures" / "dev_chavan_agent2.json"


@pytest.fixture(scope="module")
def dev_brief():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return b.build_brief(data["lab_results"], data["patient_info"],
                         data["report_notes"], data["extraction_degraded"])


def test_fallback_names_every_abnormal_row(dev_brief):
    draft = n.render_fallback(dev_brief, e.escalate(dev_brief))
    joined = " ".join(draft.key_findings)
    for a in dev_brief.abnormalities:
        assert a.test_name in joined


def test_fallback_states_the_escalation(dev_brief):
    escalation = e.escalate(dev_brief)
    draft = n.render_fallback(dev_brief, escalation)
    assert e.LEVEL_TEXT[escalation.level] in draft.summary


def test_fallback_passes_its_own_verifier(dev_brief):
    # A narrative we wrote ourselves that cannot pass verification is a bug in
    # the narrative, not in the verifier.
    draft = n.render_fallback(dev_brief, e.escalate(dev_brief))
    report = v.verify(draft, dev_brief)
    assert report.clean, [x.detail for x in report.violations]


def test_fallback_on_an_all_normal_report():
    rows = [{"test_name": "Hemoglobin", "standard_name": "Hemoglobin",
             "observed_value": "15.3", "unit": "g/dL", "reference_range": "13 - 17",
             "flag": "GREEN", "panel": "CBC", "loinc_code": "718-7"}]
    brief = b.build_brief(rows, None, [], False)
    draft = n.render_fallback(brief, e.escalate(brief))
    assert draft.key_findings == []
    assert "outside" in draft.summary.lower()


def test_fallback_quotes_lab_notes_verbatim(dev_brief):
    draft = n.render_fallback(dev_brief, e.escalate(dev_brief))
    for note in dev_brief.lab_notes:
        assert note in draft.summary


def test_fallback_handles_an_empty_report():
    brief = b.build_brief([], None, [], False)
    draft = n.render_fallback(brief, e.escalate(brief))
    assert draft.summary
    assert draft.key_findings == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_narrative.py -q`
Expected: FAIL — `cannot import name 'narrative'`

- [ ] **Step 3: Write the fallback half of `narrative.py`**

Create `src/services/reasoning/narrative.py`:

```python
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
    interval = f" (reference interval {abnormality.range_text})" if abnormality.range_text else ""
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
        questions.append(f"What could explain the results outside the range ({names})?")
        questions.append("Do any of these results need to be repeated, and if so, when?")
        questions.append("Do these results change anything about my current care?")

    tips: List[str] = []
    if abnormal:
        tips.append("Bring this report with you to your appointment.")
        tips.append("Note any symptoms you have had recently and when they started.")

    return NarrativeDraft(
        summary=" ".join(parts),
        key_findings=findings,
        doctor_questions=questions,
        lifestyle_tips=tips,
    )
```

- [ ] **Step 4: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_narrative.py -q`
Expected: PASS

If `test_fallback_passes_its_own_verifier` fails, the fallback prose tripped a rule — most likely a number that is not in `allowed_numbers`. Fix the prose, not the verifier.

- [ ] **Step 5: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/services/reasoning/narrative.py tests/test_reasoning_narrative.py
git commit -m "Add Agent 3's deterministic fallback narrative"
```

---

## Task 7: The LLM narrative path

**Files:**
- Modify: `src/services/reasoning/narrative.py`
- Test: `tests/test_reasoning_narrative.py`

**Interfaces:**
- Consumes: `render_brief` (Task 3), `get_vision_model` and `_with_retries` from the existing extractor
- Produces: `async narrate_llm(brief, escalation, violations=None) -> NarrativeDraft` — raises on failure, never returns a partial draft

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reasoning_narrative.py`:

```python
def test_prompt_contains_the_brief_and_bans_diagnosis(dev_brief):
    prompt = n.build_prompt(dev_brief, e.escalate(dev_brief))
    assert "2130" in prompt                      # a real value from the brief
    assert "Absolute Neutrophil Count" in prompt
    assert "diagnos" in prompt.lower()           # the instruction not to
    assert "Advice to rule out viral etiology in view of leukopenia." in prompt


def test_prompt_lists_prior_violations_on_retry(dev_brief):
    prompt = n.build_prompt(dev_brief, e.escalate(dev_brief),
                            violations=("3150 appears in no row of this report",))
    assert "3150 appears in no row of this report" in prompt


def test_empty_llm_reply_is_rejected():
    assert not n.is_usable(n.NarrativeDraft())
    assert not n.is_usable(n.NarrativeDraft(summary="   "))
    assert n.is_usable(n.NarrativeDraft(summary="Your white cell count is low."))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_narrative.py -q -k "prompt or usable"`
Expected: FAIL — `module has no attribute 'build_prompt'`

- [ ] **Step 3: Add the LLM path**

Append to `src/services/reasoning/narrative.py`:

```python
_SYSTEM_RULES = """\
You are writing a plain-language explanation of one lab report for the patient
who took it. Follow these rules exactly.

1. Use only the facts in the BRIEF below. Never state a number that does not
   appear there.
2. Never name a disease, condition, or cause. Do not write "anaemia",
   "infection", "deficiency", or any similar term. Do not write "this
   indicates", "this suggests", "this means", or "you have".
3. Describe what a value is and whether it is inside or outside its range.
   Nothing more.
4. Never describe a result the brief does not list as outside its range.
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
```

Add this import at the top of the file, beside the existing ones:

```python
from config.settings import settings
```

- [ ] **Step 4: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_narrative.py -q`
Expected: PASS

- [ ] **Step 5: Confirm the module imports with no API key**

Run: `venv/Scripts/python.exe -c "from src.services.reasoning import narrative; print('ok')"`
Expected: `ok`. The `google.genai` import lives inside `narrate_llm`, so importing the module must never require a key.

- [ ] **Step 6: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 7: Confirm Review Focus item 4 is covered**

`test_empty_llm_reply_is_rejected` is in Step 1 and must pass.

- [ ] **Step 8: Commit**

```bash
git add src/services/reasoning/narrative.py tests/test_reasoning_narrative.py
git commit -m "Add Agent 3's LLM narrative path with a rejected-retry prompt"
```

---

## Task 8: Orchestration and confidence

**Files:**
- Modify: `src/services/reasoning/__init__.py`
- Test: `tests/test_reasoning_analyze.py`

**Interfaces:**
- Consumes: everything from Tasks 3–7
- Produces:
  - `ReasoningResult` dataclass with `summary, key_findings, doctor_questions, lifestyle_tips, confidence_score, escalation_level, escalation_reasons, degraded`
  - `async analyze(lab_results, patient_info=None, report_notes=None, extraction_degraded=False) -> ReasoningResult`
  - `score_confidence(brief, verification, fell_back) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reasoning_analyze.py`:

```python
"""Tests for Agent 3's orchestration and confidence score."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services import reasoning as r
from src.services.reasoning import brief as b
from src.services.reasoning import verify as v

FIXTURE = Path(__file__).parent / "fixtures" / "dev_chavan_agent2.json"


@pytest.fixture(scope="module")
def dev_data():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dev_brief(dev_data):
    return b.build_brief(dev_data["lab_results"], dev_data["patient_info"],
                         dev_data["report_notes"], False)


class _Clean:
    clean, summary_ok, violations = True, True, ()
    kept: dict = {}


def test_confidence_is_between_zero_and_one(dev_brief):
    score = r.score_confidence(dev_brief, _Clean(), fell_back=False)
    assert 0.0 <= score <= 1.0


def test_degraded_extraction_halves_confidence(dev_brief, dev_data):
    degraded = b.build_brief(dev_data["lab_results"], dev_data["patient_info"],
                             dev_data["report_notes"], True)
    clean_score = r.score_confidence(dev_brief, _Clean(), fell_back=False)
    degraded_score = r.score_confidence(degraded, _Clean(), fell_back=False)
    assert degraded_score == pytest.approx(clean_score * 0.5, abs=0.01)


def test_falling_back_lowers_confidence(dev_brief):
    full = r.score_confidence(dev_brief, _Clean(), fell_back=False)
    fell = r.score_confidence(dev_brief, _Clean(), fell_back=True)
    assert fell < full


def test_empty_report_scores_zero():
    empty = b.build_brief([], None, [], False)
    assert r.score_confidence(empty, _Clean(), fell_back=False) == 0.0


@pytest.mark.asyncio
async def test_analyze_without_llm_produces_a_verified_analysis(dev_data, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr("src.services.reasoning.narrative.narrate_llm", boom)

    result = await r.analyze(dev_data["lab_results"], dev_data["patient_info"],
                             dev_data["report_notes"], False)

    assert result.degraded is True
    assert result.escalation_level == "see_doctor_promptly"
    assert result.summary
    assert result.key_findings
    assert 0.0 < result.confidence_score < 1.0


@pytest.mark.asyncio
async def test_analyze_on_an_empty_report_is_safe():
    result = await r.analyze([], None, [], False)
    assert result.escalation_level == "routine"
    assert result.confidence_score == 0.0
    assert result.key_findings == []


@pytest.mark.asyncio
async def test_analyze_drops_unverifiable_claims(dev_data, monkeypatch):
    from src.services.reasoning.narrative import NarrativeDraft

    async def poisoned(*args, **kwargs):
        return NarrativeDraft(
            summary="4 of 22 results are outside their reference interval.",
            key_findings=["Your creatinine is normal.",
                          "Total WBC Count is 2130."],
        )

    monkeypatch.setattr("src.services.reasoning.narrative.narrate_llm", poisoned)

    result = await r.analyze(dev_data["lab_results"], dev_data["patient_info"],
                             dev_data["report_notes"], False)
    assert "creatinine" not in " ".join(result.key_findings).lower()
    assert any("2130" in f for f in result.key_findings)
```

- [ ] **Step 2: Check the async test dependency**

Run: `venv/Scripts/python.exe -c "import pytest_asyncio; print('ok')"`

If that fails, add it:

```bash
venv/Scripts/python.exe -m pip install pytest-asyncio
echo "pytest-asyncio" >> requirements.txt
```

Then add to `pytest.ini` under `[pytest]`:

```ini
asyncio_mode = auto
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_analyze.py -q`
Expected: FAIL — `module 'src.services.reasoning' has no attribute 'score_confidence'`

- [ ] **Step 4: Write the orchestrator**

Replace `src/services/reasoning/__init__.py` with:

```python
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
            logger.warning("Agent 3 narrative failed verification twice — using template")
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
```

- [ ] **Step 5: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_reasoning_analyze.py -q`
Expected: PASS

- [ ] **Step 6: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 7: Confirm Review Focus item 2 is covered**

`test_empty_report_scores_zero` and `test_analyze_on_an_empty_report_is_safe` are
both in Step 1 and must pass.

- [ ] **Step 8: Commit**

```bash
git add src/services/reasoning/__init__.py tests/test_reasoning_analyze.py pytest.ini requirements.txt
git commit -m "Add Agent 3 orchestration and deterministic confidence scoring"
```

---

## Task 9: Wire Agent 3 into the pipeline and API

**Files:**
- Modify: `src/graph/nodes.py` (replace the `reason_and_verify_node` placeholder)
- Modify: `src/schemas/state.py`
- Modify: `src/schemas/report.py`
- Modify: `src/api/routes/reports.py` (`_build_response`)
- Test: `tests/test_pipeline_real_report.py`

**Interfaces:**
- Consumes: `analyze` from Task 8
- Produces: state keys `summary, key_findings, doctor_questions, lifestyle_tips, confidence_score, escalation_level, escalation_reasons, reasoning_degraded`; a `reasoning` block in the API response

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_real_report.py`:

```python
"""End-to-end tests over the real Dev chavan report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.routes.reports import _build_response
from src.graph.nodes import ground_node, reason_and_verify_node

FIXTURE = Path(__file__).parent / "fixtures" / "dev_chavan_agent2.json"
REAL_PDF = Path(r"D:\medical report analyzer data\Dev chavan Report.pdf")


@pytest.fixture(scope="module")
def dev_data():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


async def test_reason_node_produces_every_state_key(dev_data, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr("src.services.reasoning.narrative.narrate_llm", boom)

    out = await reason_and_verify_node({
        "lab_results": dev_data["lab_results"],
        "patient_info": dev_data["patient_info"],
        "report_notes": dev_data["report_notes"],
        "extraction_degraded": False,
    })

    for key in ("summary", "key_findings", "doctor_questions", "lifestyle_tips",
                "confidence_score", "escalation_level", "escalation_reasons",
                "reasoning_degraded", "current_step"):
        assert key in out
    assert out["escalation_level"] == "see_doctor_promptly"
    assert out["current_step"] == "reasoning_complete"


async def test_reason_node_never_raises(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("brief builder blew up")

    monkeypatch.setattr("src.services.reasoning.build_brief", explode)

    out = await reason_and_verify_node({"lab_results": [{"test_name": "X",
                                                         "flag": "GREEN"}]})
    assert out["current_step"] == "reasoning_failed"
    assert out["errors"]


def test_build_response_carries_the_reasoning_block(dev_data):
    response = _build_response(
        {
            "extracted_items": [{"test_name": "x"}],
            "lab_results": dev_data["lab_results"],
            "summary": "A summary.",
            "key_findings": ["a finding"],
            "doctor_questions": ["a question"],
            "lifestyle_tips": ["a tip"],
            "confidence_score": 0.82,
            "escalation_level": "see_doctor_promptly",
            "escalation_reasons": [{"rule_id": "anc_moderate",
                                    "test_name": "ANC", "detail": "x"}],
            "reasoning_degraded": False,
        },
        "rid", "Dev chavan Report.pdf", "pdf",
    )
    assert response["summary"] == "A summary."
    assert response["reasoning"]["escalation_level"] == "see_doctor_promptly"
    assert response["reasoning"]["confidence_score"] == 0.82
    assert response["reasoning"]["degraded"] is False


@pytest.mark.live
async def test_full_pipeline_on_the_real_pdf():
    """Hits Gemini. Run with: pytest -m live"""
    from src.graph.pipeline import pipeline

    if not REAL_PDF.exists():
        pytest.skip(f"report not present at {REAL_PDF}")

    state = await pipeline.ainvoke({"file_path": str(REAL_PDF), "file_type": "pdf"})

    assert state["lab_results"]
    assert state["summary"]
    assert state["escalation_level"] in (
        "discuss_at_next_visit", "see_doctor_promptly", "seek_care_now",
    )
    assert 0.0 < state["confidence_score"] <= 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_real_report.py -q`
Expected: FAIL — `reason_and_verify_node` returns only `current_step`.

- [ ] **Step 3: Add the state fields**

In `src/schemas/state.py`, replace the Node 3 block:

```python
    # --- Reasoning & Verification (Node 3 output) ---
    summary: str
    key_findings: List[str]
    doctor_questions: List[str]
    lifestyle_tips: List[str]
    confidence_score: float
    escalation_level: str            # routine | discuss_at_next_visit |
                                     # see_doctor_promptly | seek_care_now
    escalation_reasons: List[Dict[str, str]]
    reasoning_degraded: bool         # True when the templated fallback was used
```

- [ ] **Step 4: Add the report schema fields**

In `src/schemas/report.py`, add to `ReportAnalysis` after `confidence_score`:

```python
    escalation_level: str = "routine"
    escalation_reasons: List[Dict[str, str]] = Field(default_factory=list)
    reasoning_degraded: bool = False
```

- [ ] **Step 5: Implement the node**

In `src/graph/nodes.py`, replace the `reason_and_verify_node` placeholder with:

```python
async def reason_and_verify_node(state: PipelineState) -> Dict[str, Any]:
    """
    **Agent 3 · Reasoning & Verification**

    Turns Agent 2's coded, flagged rows into a plain-language analysis whose
    every factual claim has been checked in code against those rows. The LLM
    writes the prose; it does not decide what is true. When it is unavailable
    the analysis is rendered from templates instead of being dropped.

    State consumed
    ──────────────
    ``lab_results``, ``patient_info``, ``report_notes``, ``extraction_degraded``

    State produced
    ──────────────
    ``summary``, ``key_findings``, ``doctor_questions``, ``lifestyle_tips``,
    ``confidence_score``, ``escalation_level``, ``escalation_reasons``,
    ``reasoning_degraded``, ``current_step``
    """
    from src.services.reasoning import analyze

    rows = state.get("lab_results") or []
    logger.info("🧠 [reason] Starting — %d rows", len(rows))

    try:
        result = await analyze(
            rows,
            state.get("patient_info"),
            state.get("report_notes"),
            bool(state.get("extraction_degraded")),
        )
    except Exception as exc:
        logger.error("❌ [reason] Failed: %s", exc, exc_info=True)
        return {
            "summary": "",
            "key_findings": [],
            "doctor_questions": [],
            "lifestyle_tips": [],
            "confidence_score": 0.0,
            "escalation_level": "routine",
            "escalation_reasons": [],
            "reasoning_degraded": True,
            "current_step": "reasoning_failed",
            "errors": [f"Reasoning error: {exc}"],
        }

    logger.info(
        "🧠 [reason] Done — escalation=%s  confidence=%.2f  degraded=%s",
        result.escalation_level, result.confidence_score, result.degraded,
    )
    return {
        "summary": result.summary,
        "key_findings": result.key_findings,
        "doctor_questions": result.doctor_questions,
        "lifestyle_tips": result.lifestyle_tips,
        "confidence_score": result.confidence_score,
        "escalation_level": result.escalation_level,
        "escalation_reasons": result.escalation_reasons,
        "reasoning_degraded": result.degraded,
        "current_step": "reasoning_complete",
    }
```

Update the module docstring's "Upcoming" list — all three nodes are now implemented.

- [ ] **Step 6: Extend `_build_response`**

In `src/api/routes/reports.py`, add before the `"errors"` key in the returned dict:

```python
        # Agent 3 output — narrative, urgency and self-assessed confidence.
        "summary": result.get("summary", ""),
        "key_findings": result.get("key_findings", []),
        "doctor_questions": result.get("doctor_questions", []),
        "lifestyle_tips": result.get("lifestyle_tips", []),
        "reasoning": {
            "escalation_level": result.get("escalation_level", "routine"),
            "escalation_reasons": result.get("escalation_reasons", []),
            "degraded": bool(result.get("reasoning_degraded")),
            "confidence_score": result.get("confidence_score", 0.0),
        },
```

- [ ] **Step 7: Run the tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_pipeline_real_report.py -q`
Expected: PASS (the `live` test is deselected).

- [ ] **Step 8: Run the whole suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/graph/nodes.py src/schemas/state.py src/schemas/report.py \
        src/api/routes/reports.py tests/test_pipeline_real_report.py
git commit -m "Wire Agent 3 into the pipeline, state and API response"
```

---

## Task 10: Render the analysis in the UI

**Files:**
- Modify: `src/static/index.html`

**Interfaces:**
- Consumes: the `summary`, `key_findings`, `doctor_questions`, `lifestyle_tips` and `reasoning` keys added in Task 9

- [ ] **Step 1: Add the escalation banner styles**

In the `<style>` block of `src/static/index.html`, add:

```css
    .escalation { border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 1rem;
                  font-size: 0.86rem; border-left: 4px solid; }
    .escalation-routine { background: #0f2417; border-color: #2ea043; color: #7ee2a8; }
    .escalation-discuss_at_next_visit { background: #2a2411; border-color: #d29922;
                                        color: #e3b341; }
    .escalation-see_doctor_promptly { background: #2d1c10; border-color: #db6d28;
                                      color: #f0883e; }
    .escalation-seek_care_now { background: #2d1418; border-color: #f85149;
                                color: #ff7b72; }
    .analysis-block { margin-bottom: 1.1rem; }
    .analysis-block h4 { font-size: 0.8rem; text-transform: uppercase;
                         letter-spacing: 0.05em; color: var(--text-muted);
                         margin: 0 0 0.4rem; }
    .analysis-block ul { margin: 0; padding-left: 1.1rem; }
    .analysis-block li { margin-bottom: 0.3rem; font-size: 0.86rem; }
    .analysis-summary { font-size: 0.92rem; line-height: 1.6; }
    .confidence-note { font-size: 0.76rem; color: var(--text-muted);
                       margin-top: 0.6rem; }
```

- [ ] **Step 2: Add the container**

In `src/static/index.html`, immediately after the `extractionBanner` div (around line 977), insert:

```html
          <!-- Agent 3 output -->
          <div id="analysisSection" style="display:none;"></div>
```

- [ ] **Step 3: Add the renderer**

Add this function beside the other render helpers:

```javascript
    // Agent 3's narrative. Every claim here has already been checked in code
    // against the lab rows, so it is rendered as given — but escapeHtml still
    // applies, because the text originates from a model.
    function renderAnalysis(data) {
      const host = document.getElementById('analysisSection');
      const reasoning = data.reasoning || {};
      const hasAnalysis = data.summary || (data.key_findings || []).length;

      if (!hasAnalysis) {
        host.style.display = 'none';
        host.innerHTML = '';
        return;
      }

      const level = reasoning.escalation_level || 'routine';
      const levelLabel = {
        routine: 'Nothing outside range',
        discuss_at_next_visit: 'Raise at your next appointment',
        see_doctor_promptly: 'Show a doctor soon',
        seek_care_now: 'Contact a doctor today',
      }[level] || level;

      const list = (title, items) => {
        if (!items || !items.length) return '';
        return `
          <div class="analysis-block">
            <h4>${title}</h4>
            <ul>${items.map(i => `<li>${escapeHtml(i)}</li>`).join('')}</ul>
          </div>`;
      };

      const confidence = Math.round((reasoning.confidence_score || 0) * 100);
      const degradedNote = reasoning.degraded
        ? ' Written from a template because the language model was unavailable.'
        : '';

      host.innerHTML = `
        <div class="escalation escalation-${level}"><strong>${levelLabel}</strong></div>
        <div class="analysis-block">
          <h4>Summary</h4>
          <div class="analysis-summary">${escapeHtml(data.summary || '')}</div>
        </div>
        ${list('Key findings', data.key_findings)}
        ${list('Questions for your doctor', data.doctor_questions)}
        ${list('Before your appointment', data.lifestyle_tips)}
        <div class="confidence-note">
          Confidence ${confidence}%.${degradedNote}
          This is not a medical diagnosis.
        </div>`;
      host.style.display = 'block';
    }
```

- [ ] **Step 4: Call it**

In `renderResults`, after the `renderGroundingSummary(data.grounding);` line, add:

```javascript
      renderAnalysis(data);
```

- [ ] **Step 5: Verify in the browser**

```bash
venv/Scripts/python.exe main.py
```

Open `http://localhost:8000`, upload `D:\medical report analyzer data\Dev chavan Report.pdf`, and confirm: an orange "Show a doctor soon" banner, a summary naming the low white cell count, key findings listing the abnormal rows, and a confidence percentage. Stop the server when done.

- [ ] **Step 6: Commit**

```bash
git add src/static/index.html
git commit -m "Render Agent 3's analysis and escalation banner"
```

---

## Task 11: Live end-to-end verification

**Files:**
- None modified. This task runs what earlier tasks built.

- [ ] **Step 1: Run the full offline suite**

Run: `venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no warnings in the output.

- [ ] **Step 2: Run the live test**

Run: `venv/Scripts/python.exe -m pytest -m live -q -s`

Gemini returns 503 intermittently. Retry up to three times. If it stays unavailable, record that in the final report rather than claiming the live path is verified — an untested path is untested.

- [ ] **Step 3: Read the live output and judge it**

Print the analysis and read it as a patient would:

```bash
venv/Scripts/python.exe -c "
import asyncio, json, sys
sys.path.insert(0, '.')
from src.graph.pipeline import pipeline
state = asyncio.run(pipeline.ainvoke({
    'file_path': r'D:\medical report analyzer data\Dev chavan Report.pdf',
    'file_type': 'pdf'}))
print(json.dumps({k: state.get(k) for k in
    ('summary','key_findings','doctor_questions','lifestyle_tips',
     'escalation_level','escalation_reasons','confidence_score',
     'reasoning_degraded')}, indent=2))
"
```

Check by hand:
- No condition is named anywhere in the output.
- Every number in the summary appears in the report.
- The escalation level is `see_doctor_promptly`.
- The lab's note is quoted word-for-word if it is quoted at all.

Any failure here is a real defect. Write the failing case as a test in the module that owns it, then fix it.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "Fix defects found in live end-to-end verification"
```

---

## Self-Review

**Spec coverage.** Module layout → Tasks 3–8. Brief → Task 3. Escalation → Task 4. Narrative and fallback → Tasks 6, 7. Verifier → Task 5. Confidence → Task 8. Wiring → Task 9. UI → Task 10. Testing → every task, plus Task 11. AMBER band fix → Task 1. Fixture → Task 2. No spec section is unimplemented.

**Spec deviations, both deliberate:**
1. Escalation matches on `loinc_code`, not `ontology_key`, because `process_item` does not emit `ontology_key`. Documented under Global Constraints.
2. `score_confidence` lives in `__init__.py` rather than its own module, since combining signals is the orchestrator's job.

**Type consistency.** `ClinicalBrief`, `Abnormality`, `FiredRule`, `Escalation`, `Violation`, `VerificationReport`, `NarrativeDraft` and `ReasoningResult` are each defined once and used with the same field names throughout. `build_brief`, `render_brief`, `normalise_number`, `escalate`, `verify`, `render_fallback`, `build_prompt`, `is_usable`, `narrate_llm`, `analyze` and `score_confidence` keep one signature each.

**Review Focus coverage.** Item 1 → Task 5 Steps 1, 7. Item 2 → Task 8 Steps 1, 7. Item 3 → Task 3 Steps 1, 7. Item 4 → Task 7 Steps 1, 7. Item 5 → Task 3 Steps 1, 9.
