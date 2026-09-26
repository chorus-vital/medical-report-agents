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
    assert dev_brief.total_rows == 25
    assert sum(dev_brief.flag_counts.values()) == dev_brief.total_rows


def test_red_abnormalities_come_before_amber(dev_brief):
    flags = [a.flag for a in dev_brief.abnormalities]
    assert flags == sorted(flags, key=lambda f: 0 if f == "RED" else 1)
    assert "GREEN" not in flags
    assert "UNKNOWN" not in flags


def test_abnormality_records_direction(dev_brief):
    # The brief prefers Agent 2's canonical standard_name ("Total WBC Count")
    # over the name the lab printed ("Total White Blood Cell Count (TC)").
    wbc = next(a for a in dev_brief.abnormalities if a.loinc_code == "6690-2")
    assert wbc.test_name == "Total WBC Count"
    assert wbc.value == "2130"
    assert wbc.direction == "below"
    assert wbc.flag == "RED"
    assert wbc.loinc_code == "6690-2"


def test_allowed_numbers_hold_values_and_bounds(dev_brief):
    assert "2130" in dev_brief.allowed_numbers    # an observed value
    assert "4000" in dev_brief.allowed_numbers    # a range bound
    assert "25" in dev_brief.allowed_numbers      # a count
    assert "9999" not in dev_brief.allowed_numbers


def test_allowed_analytes_hold_only_tested_rows(dev_brief):
    assert "hemoglobin" in dev_brief.allowed_analytes
    assert "creatinine" not in dev_brief.allowed_analytes  # never tested here


def test_allowed_loinc_holds_every_coded_row(dev_brief):
    assert "6690-2" in dev_brief.allowed_loinc     # Total WBC
    assert "751-8" in dev_brief.allowed_loinc      # ANC
    assert "2160-0" not in dev_brief.allowed_loinc  # creatinine, never tested


def test_unknown_analytes_name_only_unevaluated_rows(dev_brief):
    # Agent 2 could not evaluate the two indices or the smear.
    assert "mentzer index" in dev_brief.unknown_analytes
    assert "peripheral smear for malarial parasite" in dev_brief.unknown_analytes
    # A GREEN row is evaluated, so it must NOT be in here — otherwise calling a
    # normal haemoglobin "normal" would be reported as a violation.
    assert "hemoglobin" not in dev_brief.unknown_analytes


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


def test_render_brief_carries_the_lab_note(dev_brief):
    assert dev_brief.lab_notes[0] in b.render_brief(dev_brief)
