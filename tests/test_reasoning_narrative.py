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


def test_fallback_flags_a_degraded_extraction():
    rows = [{"test_name": "Hemoglobin", "standard_name": "Hemoglobin",
             "observed_value": "15.3", "unit": "g/dL", "reference_range": "13 - 17",
             "flag": "GREEN", "panel": "CBC", "loinc_code": "718-7"}]
    brief = b.build_brief(rows, None, [], True)
    draft = n.render_fallback(brief, e.escalate(brief))
    assert "incomplete" in draft.summary.lower()
