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


# ───────────── Post-merge review findings (C2, I1, I2, I3, I4, I9) ────────────


def test_fabricated_value_for_a_real_analyte_is_caught(brief):
    # C2: small integers used to be exempt, which covers the whole reporting
    # scale of haemoglobin, potassium, calcium and bilirubin. The real row is
    # 15.3; claiming 9 must not pass.
    draft = Draft(summary="Your haemoglobin is 9 g/dL.")
    report = v.verify(draft, brief)
    assert not report.summary_ok
    assert any(x.kind == "number" for x in report.violations)


def test_untested_acronym_analyte_is_caught(brief):
    # I1: aliases under five characters were skipped entirely, so the model's
    # own shorthand walked straight through.
    draft = Draft(key_findings=["Your TSH is 4 and your ESR is 3."])
    report = v.verify(draft, brief)
    assert report.kept["key_findings"] == []


def test_a_lowercase_english_word_is_not_mistaken_for_an_acronym(brief):
    # The acronym scan must not fire on ordinary prose containing, say, "alt".
    draft = Draft(key_findings=["There is an alternative worth discussing."])
    report = v.verify(draft, brief)
    assert not [x for x in report.violations if x.kind == "analyte"]


def test_restating_a_lab_note_as_your_own_conclusion_is_caught(brief):
    # I2: the exemption was span-based, so the lab's hedged "rule out" could be
    # replayed unattributed as a conclusion.
    note = brief.lab_notes[0]
    draft = Draft(summary=f"{note} That is what your results show.")
    report = v.verify(draft, brief)
    assert not report.summary_ok


def test_a_quoted_lab_note_is_still_exempt(brief):
    note = brief.lab_notes[0]
    draft = Draft(summary=f'Your lab printed this note: "{note}"')
    report = v.verify(draft, brief)
    assert report.summary_ok, [x.detail for x in report.violations]


def test_a_short_lab_note_cannot_disable_a_blocklist_term():
    # I3: report_notes comes from Agent 1 reading the PDF. A one-word note like
    # a section heading used to switch that term off for the whole narrative.
    rows = [{"test_name": "Hemoglobin", "standard_name": "Hemoglobin",
             "observed_value": "15.3", "unit": "g/dL", "reference_range": "13 - 17",
             "flag": "GREEN", "panel": "CBC", "loinc_code": "718-7"}]
    short_note_brief = b.build_brief(rows, None, ["infection"], False)
    draft = Draft(summary="There are signs of infection here.")
    assert not v.verify(draft, short_note_brief).summary_ok


def test_a_spelled_out_decimal_is_caught(brief):
    # I4: the retry prompt names the offending number, which invites the model
    # to spell it out instead.
    draft = Draft(summary="Your haemoglobin is nine point two grams per decilitre.")
    assert not v.verify(draft, brief).summary_ok


def test_ordinary_number_words_are_not_flagged(brief):
    draft = Draft(key_findings=["One of your results is outside its range."])
    assert not [x for x in v.verify(draft, brief).violations if x.kind == "number"]


@pytest.mark.parametrize(
    "phrasing",
    [
        "Your Malarial Parasite result is normal.",
        "Your Malarial Parasite result is unremarkable.",
        "Your Malarial Parasite result is within the reference interval.",
        "Your Malarial Parasite result is not elevated.",
    ],
)
def test_unknown_row_cannot_be_called_normal_however_it_is_phrased(brief, phrasing):
    # I9: the normality vocabulary was narrow enough to sidestep by rewording.
    report = v.verify(Draft(key_findings=[phrasing]), brief)
    assert any(x.kind == "unknown_row" for x in report.violations), phrasing


def test_an_ontology_gloss_is_exempt_from_the_blocklist():
    # "The number of infection-fighting white blood cells" is ours, from the
    # ontology — vetted repo content, not a model claim. It must not be read as
    # the model diagnosing an infection.
    rows = [{"test_name": "Total WBC Count", "standard_name": "Total WBC Count",
             "observed_value": "2130", "unit": "cells/mm3",
             "reference_range": "4000 - 10000", "flag": "RED", "panel": "CBC",
             "loinc_code": "6690-2",
             "explanation": "The number of infection-fighting white blood cells."}]
    gloss_brief = b.build_brief(rows, None, [], False)
    gloss = gloss_brief.abnormalities[0].plain_meaning
    draft = Draft(key_findings=[f"Total WBC Count is 2130. {gloss}"])
    report = v.verify(draft, gloss_brief)
    assert report.kept["key_findings"], [x.detail for x in report.violations]


def test_the_model_still_cannot_claim_an_infection_in_its_own_words():
    rows = [{"test_name": "Total WBC Count", "standard_name": "Total WBC Count",
             "observed_value": "2130", "unit": "cells/mm3",
             "reference_range": "4000 - 10000", "flag": "RED", "panel": "CBC",
             "loinc_code": "6690-2",
             "explanation": "The number of infection-fighting white blood cells."}]
    gloss_brief = b.build_brief(rows, None, [], False)
    draft = Draft(summary="This low count is a sign of infection.")
    assert not v.verify(draft, gloss_brief).summary_ok


@pytest.mark.parametrize(
    "text",
    [
        "White blood cells that fight infection are low.",
        "These are infection-fighting cells.",
        "These are infection‑fighting cells.",   # non-breaking hyphen
        "Cells that defend against infection.",
        "Hemoglobin is 15.3, and these cells help protect against infection.",
    ],
)
def test_a_condition_word_used_as_a_definition_is_allowed(brief, text):
    # Describing what a cell DOES is not diagnosing the patient with it.
    report = v.verify(Draft(key_findings=[text]), brief)
    assert report.kept["key_findings"] == [text], [x.detail for x in report.violations]


@pytest.mark.parametrize(
    "text",
    [
        "This low count is a sign of infection.",
        "Your results are consistent with infection.",
        "This is likely an infection.",
        "You have an infection.",
        "These results indicate anaemia.",
    ],
)
def test_a_condition_word_predicated_of_the_patient_is_still_caught(brief, text):
    assert not v.verify(Draft(summary=text), brief).summary_ok
