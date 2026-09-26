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
    assert report.clean, [x.detail for x in report.violations]
    assert report.violations == ()


def test_fabricated_number_is_caught(brief):
    draft = Draft(summary="Your white blood cell count is 3150.")
    report = v.verify(draft, brief)
    assert not report.clean
    assert any(x.kind == "number" and "3150" in x.detail for x in report.violations)


def test_untested_analyte_is_caught(brief):
    draft = Draft(key_findings=["Your creatinine is within the expected interval."])
    report = v.verify(draft, brief)
    assert any(x.kind == "analyte" for x in report.violations)
    assert report.kept["key_findings"] == []


def test_a_synonym_for_a_tested_analyte_is_allowed(brief):
    # "haemoglobin" is an ontology alias for a row this report DID carry.
    draft = Draft(key_findings=["Your haemoglobin is 15.3, inside 13 to 17."])
    report = v.verify(draft, brief)
    assert not [x for x in report.violations if x.kind == "analyte"]


def test_diagnosis_language_is_caught(brief):
    draft = Draft(summary="This indicates anaemia and you have an infection.")
    report = v.verify(draft, brief)
    assert any(x.kind == "diagnosis" for x in report.violations)
    assert not report.summary_ok


def test_claim_about_an_unknown_row_is_caught(brief):
    draft = Draft(key_findings=["Your Malarial Parasite result is normal."])
    report = v.verify(draft, brief)
    assert any(x.kind == "unknown_row" for x in report.violations)


def test_calling_a_green_row_normal_is_allowed(brief):
    # Hemoglobin was evaluated and is GREEN. Saying so must not be a violation.
    draft = Draft(key_findings=["Your Hemoglobin of 15.3 is normal."])
    report = v.verify(draft, brief)
    assert not [x for x in report.violations if x.kind == "unknown_row"]


def test_only_the_offending_bullet_is_dropped(brief):
    draft = Draft(
        summary="Your white blood cell count is 2130.",
        key_findings=[
            "Total WBC Count 2130 is below the reference interval",
            "Your creatinine is 1.1.",
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


def test_empty_draft_produces_no_violations(brief):
    report = v.verify(Draft(), brief)
    assert report.violations == ()
