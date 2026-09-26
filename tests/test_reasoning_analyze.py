"""Tests for Agent 3's orchestration and confidence score."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services import reasoning as r
from src.services.reasoning import brief as b

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


def test_unknown_rows_lower_confidence():
    coded = [{"test_name": "Hemoglobin", "standard_name": "Hemoglobin",
              "observed_value": "15.3", "unit": "g/dL", "reference_range": "13 - 17",
              "flag": "GREEN", "panel": "CBC", "loinc_code": "718-7"}]
    with_unknown = coded + [{"test_name": "Odd Index", "observed_value": "1",
                             "unit": None, "reference_range": None,
                             "flag": "UNKNOWN", "panel": None, "loinc_code": None}]
    a = r.score_confidence(b.build_brief(coded, None, [], False), _Clean(), False)
    c = r.score_confidence(b.build_brief(with_unknown, None, [], False), _Clean(), False)
    assert c < a


def test_empty_report_scores_zero():
    empty = b.build_brief([], None, [], False)
    assert r.score_confidence(empty, _Clean(), fell_back=False) == 0.0


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


async def test_analyze_on_an_empty_report_is_safe():
    result = await r.analyze([], None, [], False)
    assert result.escalation_level == "no_data"
    assert result.confidence_score == 0.0
    assert result.key_findings == []


async def test_analyze_drops_unverifiable_claims(dev_data, monkeypatch):
    from src.services.reasoning.narrative import NarrativeDraft

    async def poisoned(*args, **kwargs):
        return NarrativeDraft(
            summary="6 of 25 results are outside their reference interval.",
            key_findings=["Your creatinine is 1.1.",
                          "Total WBC Count is 2130."],
        )

    monkeypatch.setattr("src.services.reasoning.narrative.narrate_llm", poisoned)

    result = await r.analyze(dev_data["lab_results"], dev_data["patient_info"],
                             dev_data["report_notes"], False)
    assert "creatinine" not in " ".join(result.key_findings).lower()
    assert any("2130" in f for f in result.key_findings)
    assert result.degraded is False


async def test_analyze_retries_once_then_falls_back(dev_data, monkeypatch):
    from src.services.reasoning.narrative import NarrativeDraft

    calls = []

    async def always_bad(brief, escalation, violations=()):
        calls.append(violations)
        return NarrativeDraft(summary="This indicates anaemia.")

    monkeypatch.setattr("src.services.reasoning.narrative.narrate_llm", always_bad)

    result = await r.analyze(dev_data["lab_results"], dev_data["patient_info"],
                             dev_data["report_notes"], False)

    assert len(calls) == 2              # one attempt, then one retry
    assert calls[1]                     # the retry was told what was wrong
    assert result.degraded is True      # then the template took over
    assert "anaemia" not in result.summary.lower()


async def test_analyze_carries_escalation_reasons(dev_data, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("503")

    monkeypatch.setattr("src.services.reasoning.narrative.narrate_llm", boom)
    result = await r.analyze(dev_data["lab_results"], dev_data["patient_info"],
                             dev_data["report_notes"], False)
    ids = [reason["rule_id"] for reason in result.escalation_reasons]
    assert "anc_moderate" in ids
    for reason in result.escalation_reasons:
        assert set(reason) == {"rule_id", "test_name", "detail"}


# ──────────── Post-merge review findings (I5, I6, I8) ────────────


def test_confidence_falls_when_the_narrative_ignores_the_abnormalities(dev_brief):
    # I5: the score measured provenance of the rows and whether anything was
    # dropped, never whether the narrative said anything. A one-line summary
    # with no findings scored 0.88 and was shown as "Confidence 88%".
    covered = " ".join(a.test_name for a in dev_brief.abnormalities)
    thorough = r.score_confidence(dev_brief, _Clean(), False, narrative=covered)
    vacuous = r.score_confidence(dev_brief, _Clean(), False,
                                 narrative="Your results are ready to review.")
    assert vacuous < thorough
    assert vacuous < 0.5


def test_coverage_does_not_penalise_an_all_normal_report():
    rows = [{"test_name": "Hemoglobin", "standard_name": "Hemoglobin",
             "observed_value": "15.3", "unit": "g/dL", "reference_range": "13 - 17",
             "flag": "GREEN", "panel": "CBC", "loinc_code": "718-7"}]
    brief = b.build_brief(rows, None, [], False)
    assert r.score_confidence(brief, _Clean(), False, narrative="All normal.") == 1.0


async def test_empty_report_is_not_presented_as_nothing_abnormal():
    # I6: escalation "routine" renders a green "Nothing outside range" banner.
    # No data is not the same as no disease.
    result = await r.analyze([], None, [], False)
    assert result.escalation_level == "no_data"
    assert result.confidence_score == 0.0


async def test_dropped_claims_are_counted(dev_data, monkeypatch):
    # I8: bullets removed by the verifier vanished with no marker and no count.
    from src.services.reasoning.narrative import NarrativeDraft

    async def poisoned(*args, **kwargs):
        return NarrativeDraft(
            summary="6 of 25 results are outside their reference interval.",
            key_findings=["Your creatinine is 1.1.",
                          "Your TSH is 4.",
                          "Total WBC Count is 2130."],
        )

    monkeypatch.setattr("src.services.reasoning.narrative.narrate_llm", poisoned)
    result = await r.analyze(dev_data["lab_results"], dev_data["patient_info"],
                             dev_data["report_notes"], False)
    assert result.dropped_claims == 2
    assert len(result.key_findings) == 1
