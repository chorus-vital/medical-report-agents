"""End-to-end tests over the real Dev chavan report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.routes.reports import _build_response
from src.graph.nodes import reason_and_verify_node

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


async def test_reason_node_on_no_rows():
    out = await reason_and_verify_node({"lab_results": []})
    assert out["current_step"] == "reasoning_complete"
    assert out["confidence_score"] == 0.0
    assert out["escalation_level"] == "routine"


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
    assert response["key_findings"] == ["a finding"]
    assert response["reasoning"]["escalation_level"] == "see_doctor_promptly"
    assert response["reasoning"]["confidence_score"] == 0.82
    assert response["reasoning"]["degraded"] is False


def test_build_response_defaults_when_agent_3_did_not_run():
    response = _build_response(
        {"extracted_items": [{"test_name": "x"}], "lab_results": []},
        "rid", "f.pdf", "pdf",
    )
    assert response["summary"] == ""
    assert response["reasoning"]["escalation_level"] == "routine"


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
