"""
Tests for the extraction service.

The offline tests cover every pure function in the extraction path — text
repair, chunking, cleaning and the regex fallback — so accuracy regressions are
caught without spending API quota.

The live tests (marked ``live``) actually call Gemini. Run them with::

    pytest -m live
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.services.extractor import (
    _canon_unit,
    _chunk_pages,
    _clean_items,
    _extract_stacked_layout,
    _nullify,
    _parse_json_text,
    _regex_fallback,
    _regex_patient_info,
    detect_file_type,
)
from src.services.pdf_text import repair_mojibake

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "data" / "samples"


# ───────────────────────── File-type detection ──────────────────────────────


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("report.pdf", "pdf"),
        ("REPORT.PDF", "pdf"),
        ("scan.jpg", "image"),
        ("scan.jpeg", "image"),
        ("scan.png", "image"),
        ("scan.webp", "image"),
        ("notes.docx", "docx"),
        ("data.txt", "text"),
        ("data.csv", "text"),
    ],
)
def test_detect_file_type(filename, expected):
    assert detect_file_type(filename) == expected


def test_detect_file_type_rejects_unknown():
    with pytest.raises(ValueError):
        detect_file_type("report.xlsx")


# ─────────────────────────── Mojibake repair ────────────────────────────────
#
# Real lab PDFs embed subsetted fonts whose superscripts and micro-signs lose
# their Unicode mapping. Without repair these land in the unit field as U+FFFD
# and every downstream unit comparison fails.


@pytest.mark.parametrize(
    "broken,expected",
    [
        ("mill/mm�", "mill/mm³"),
        ("cells/mm�", "cells/mm³"),
        ("10�/�L", "10³/µL"),
        ("mm\x00", "mm³"),
        ("�mol/L", "µmol/L"),
    ],
)
def test_repair_mojibake_units(broken, expected):
    assert repair_mojibake(broken) == expected


def test_repair_mojibake_leaves_clean_text_alone():
    clean = "Hemoglobin (Hb) 15.3 g/dL 13 - 17"
    assert repair_mojibake(clean) == clean


def test_repair_mojibake_drops_stray_control_chars():
    # Unmapped glyphs become spaces, never control characters.
    assert "\x00" not in repair_mojibake("Red Blood Cells \x00RBC\x00 Count")


# ───────────────────────────── Chunking ─────────────────────────────────────


def test_chunk_pages_keeps_small_documents_whole():
    assert len(_chunk_pages(["short page one", "short page two"])) == 1


def test_chunk_pages_splits_when_over_budget():
    pages = ["x" * 8000, "y" * 8000, "z" * 8000]
    chunks = _chunk_pages(pages)
    assert len(chunks) > 1
    # Nothing may be dropped: the old code truncated at 15k characters and
    # silently lost every result past it.
    assert sum(c.count("x") for c in chunks) == 8000
    assert sum(c.count("z") for c in chunks) == 8000


def test_chunk_pages_splits_a_single_oversized_page():
    page = "\n".join(f"line {i} " + "w" * 100 for i in range(500))
    chunks = _chunk_pages([page])
    assert len(chunks) > 1
    assert all(len(c) <= 20_000 for c in chunks)


def test_chunk_pages_ignores_blank_pages():
    assert _chunk_pages(["", "   ", "\n"]) == []


# ────────────────────────── Cleaning / dedup ────────────────────────────────


def test_clean_items_deduplicates_repeated_rows():
    items = [
        {"test_name": "Hemoglobin", "observed_value": "15.3"},
        {"test_name": "hemoglobin", "observed_value": "15.3"},
    ]
    assert len(_clean_items(items)) == 1


def test_clean_items_keeps_same_test_with_different_values():
    items = [
        {"test_name": "Glucose", "observed_value": "118"},
        {"test_name": "Glucose", "observed_value": "142"},
    ]
    assert len(_clean_items(items)) == 2


def test_clean_items_drops_headers_and_footers():
    items = [
        {"test_name": "Test", "observed_value": "Result"},
        {"test_name": "Page 2 of 5", "observed_value": "x"},
        {"test_name": "Patient ID", "observed_value": "OHPRN2D72350218"},
        {"test_name": "Approved By", "observed_value": "Dr. Aekta"},
        {"test_name": "Hemoglobin", "observed_value": "15.3"},
    ]
    cleaned = _clean_items(items)
    assert [i["test_name"] for i in cleaned] == ["Hemoglobin"]


def test_clean_items_requires_a_value():
    assert _clean_items([{"test_name": "Hemoglobin", "observed_value": ""}]) == []


def test_clean_items_collapses_multiline_reference_range():
    items = [
        {
            "test_name": "Mentzer Index",
            "observed_value": "14.8",
            "reference_range": "Beta Thalassemia trait: < 14\nIron deficiency anaemia: >= 14",
        }
    ]
    ref = _clean_items(items)[0]["reference_range"]
    assert "\n" not in ref
    assert ref == "Beta Thalassemia trait: < 14; Iron deficiency anaemia: >= 14"


def test_clean_items_repairs_units():
    items = [{"test_name": "RBC", "observed_value": "5.47", "unit": "mill/mm�"}]
    assert _clean_items(items)[0]["unit"] == "mill/mm³"


@pytest.mark.parametrize(
    "raw,expected", [("ul", "µL"), ("10^3/uL", "10³/µL"), ("g/dL", "g/dL"), ("", None)]
)
def test_canon_unit(raw, expected):
    assert _canon_unit(raw) == expected


@pytest.mark.parametrize("value", ["", " ", "null", "N/A", "-", "--", "none"])
def test_nullify_sentinels(value):
    assert _nullify(value) is None


# ──────────────────────── LLM response parsing ──────────────────────────────


def test_parse_json_text_handles_markdown_fences():
    raw = '```json\n{"lab_results": [{"test_name": "Hb", "observed_value": "15.3"}]}\n```'
    parsed = _parse_json_text(raw)
    assert len(parsed.lab_results) == 1
    assert parsed.lab_results[0].test_name == "Hb"


def test_parse_json_text_handles_surrounding_prose():
    raw = 'Here you go:\n{"lab_results": [{"test_name": "Hb", "observed_value": "15.3"}]}\nHope this helps!'
    assert len(_parse_json_text(raw).lab_results) == 1


def test_parse_json_text_survives_garbage():
    parsed = _parse_json_text("I could not read that document.")
    assert parsed.lab_results == []


def test_parse_json_text_salvages_partially_invalid_rows():
    raw = '{"lab_results": [{"test_name": "Hb", "observed_value": "15.3"}, {"no_name": 1}]}'
    assert len(_parse_json_text(raw).lab_results) == 1


# ────────────────────── Regex fallback (degraded path) ──────────────────────
#
# This runs only when the LLM is unreachable. It must still recover the bulk of
# a real report — the original version returned two junk rows for the layout
# below, which is what made a missing API key look like an unparseable PDF.

STACKED_REPORT = """\
Test Result Biological Reference Interval
Haematology
Complete Blood Count (CBC) Whole Blood EDTA
Red Blood Cells (RBC) Count
DC Impedance Method
5.47 4.5 - 5.5
mill/mm³
Hemoglobin (Hb)
Cyanide-free SLS method
15.3 13 - 17
g/dL
Mean Corpuscular Volume (MCV)
Calculated
81.1 83 - 101
fL
Total White Blood Cell Count (TC)
Flow Cytometry
2130 4000 - 10000
cells/mm³
"""


def test_stacked_layout_recovers_name_value_unit_and_range():
    rows = {r["test_name"]: r for r in _extract_stacked_layout(STACKED_REPORT)}

    assert "Hemoglobin (Hb)" in rows
    hb = rows["Hemoglobin (Hb)"]
    assert hb["observed_value"] == "15.3"
    assert hb["unit"] == "g/dL"
    assert hb["reference_range"] == "13 - 17"
    assert hb["method"] == "Cyanide-free SLS method"


def test_stacked_layout_does_not_mistake_method_for_test_name():
    names = [r["test_name"] for r in _extract_stacked_layout(STACKED_REPORT)]
    assert "Calculated" not in names
    assert "Flow Cytometry" not in names
    assert "DC Impedance Method" not in names


def test_stacked_layout_does_not_absorb_the_previous_unit_line():
    # 'mill/mm³' terminates the row above; it must not become part of the next
    # test's name.
    names = [r["test_name"] for r in _extract_stacked_layout(STACKED_REPORT)]
    assert all(not n.startswith("mill/") for n in names), names


def test_regex_fallback_recovers_most_of_a_stacked_report():
    items = _clean_items(_regex_fallback(STACKED_REPORT))
    assert len(items) == 4
    assert {i["test_name"] for i in items} == {
        "Red Blood Cells (RBC) Count",
        "Hemoglobin (Hb)",
        "Mean Corpuscular Volume (MCV)",
        "Total White Blood Cell Count (TC)",
    }


ALIGNED_REPORT = """\
Hemoglobin                 12.1        g/dL          13.0 - 17.0
Total Cholesterol          242         mg/dL         < 200
"""


def test_regex_fallback_handles_whitespace_aligned_columns():
    items = _clean_items(_regex_fallback(ALIGNED_REPORT))
    by_name = {i["test_name"]: i for i in items}
    assert by_name["Hemoglobin"]["observed_value"] == "12.1"
    assert by_name["Total Cholesterol"]["reference_range"] == "< 200"


def test_regex_fallback_handles_pipe_tables():
    items = _clean_items(_regex_fallback("Hemoglobin | 14.5 | g/dL | 12.0 - 16.0"))
    assert items[0]["test_name"] == "Hemoglobin"
    assert items[0]["unit"] == "g/dL"


def test_regex_patient_info_reads_inline_demographics():
    info = _regex_patient_info(
        "Dev chavan Report Ref. ID : BLR6817411\n"
        "23 Year(s)/Male Patient ID : OHPRN2D72350218\n"
        "Collected : 02/08/2026 07:14 PM\n"
    )
    assert info["age"] == "23"
    assert info["sex"] == "Male"
    assert info["patient_id"] == "OHPRN2D72350218"
    assert info["report_ref_id"] == "BLR6817411"


def test_regex_patient_info_returns_none_when_absent():
    assert _regex_patient_info("no demographics here") is None


# ───────────────────────── Live (opt-in) tests ──────────────────────────────


@pytest.mark.live
def test_live_text_sample_extraction():
    """End-to-end extraction of the bundled text sample via the real model."""
    from src.services.extractor import extract_from_file_detailed

    sample = SAMPLES / "sample_report_01.txt"
    if not sample.exists():
        pytest.skip("sample_report_01.txt not present")

    result = asyncio.run(extract_from_file_detailed(str(sample), "text"))

    assert not result.degraded, result.warnings
    # The sample contains 39 tests across seven panels.
    assert len(result.items) >= 35, f"only got {len(result.items)}"
    assert result.patient_info["name"] == "Rahul Sharma"

    by_name = {i["test_name"].lower(): i for i in result.items}
    assert by_name["hemoglobin"]["observed_value"] == "12.1"
    assert by_name["hemoglobin"]["unit"] == "g/dL"
