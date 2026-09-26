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
                                     "2000 - 7000", "RED", "751-8")]))
    assert result.level == "see_doctor_promptly"
    assert "anc_moderate" in [r.rule_id for r in result.reasons]


# Unit scaling. A lab printing platelets as 194 against "150 - 450" is using
# x10^3/uL; comparing that raw against a /mm3 threshold of 20,000 would fire a
# seek_care_now alert on a perfectly normal count.
def test_normal_platelets_in_thousands_do_not_fire_a_critical_rule():
    result = e.escalate(_brief([_row("Platelet Count", "194", "150 - 450",
                                     "GREEN", "777-3")]))
    assert result.level == "routine"
    assert result.reasons == ()


def test_low_platelets_in_thousands_still_fire():
    result = e.escalate(_brief([_row("Platelet Count", "15", "150 - 450",
                                     "RED", "777-3")]))
    assert result.level == "seek_care_now"
    assert "platelet_critical" in [r.rule_id for r in result.reasons]


def test_anc_in_thousands_is_scaled_too():
    result = e.escalate(_brief([_row("Absolute Neutrophil Count", "0.4",
                                     "2.0 - 7.0", "RED", "751-8")]))
    assert result.level == "seek_care_now"
    assert "anc_critical" in [r.rule_id for r in result.reasons]


def test_highest_level_wins():
    rows = [
        _row("MCV", "81.1", "83 - 101", "AMBER", "787-2"),
        _row("Absolute Neutrophil Count", "400", "2000 - 7000", "RED", "751-8"),
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
    # The normal platelet count of 194 must not have fired anything.
    assert "platelet_critical" not in [r.rule_id for r in result.reasons]
    assert "platelet_low" not in [r.rule_id for r in result.reasons]
