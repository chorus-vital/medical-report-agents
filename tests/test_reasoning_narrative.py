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


def test_prompt_contains_the_brief_and_bans_diagnosis(dev_brief):
    prompt = n.build_prompt(dev_brief, e.escalate(dev_brief))
    assert "2130" in prompt                      # a real value from the brief
    assert "Absolute Neutrophil Count" in prompt
    assert "diagnos" in prompt.lower()           # the instruction not to
    assert dev_brief.lab_notes[0] in prompt


def test_prompt_lists_prior_violations_on_retry(dev_brief):
    prompt = n.build_prompt(dev_brief, e.escalate(dev_brief),
                            violations=("3150 appears in no row of this report",))
    assert "3150 appears in no row of this report" in prompt


def test_empty_llm_reply_is_rejected():
    assert not n.is_usable(n.NarrativeDraft())
    assert not n.is_usable(n.NarrativeDraft(summary="   "))
    assert n.is_usable(n.NarrativeDraft(summary="Your white cell count is low."))


# ───────────────── Provider chain: Groq first, Gemini fallback ────────────────
#
# Agent 3 sends text and gets JSON back — it has no use for the vision client it
# was originally wired to, and being pinned to Gemini meant Agent 1 and Agent 3
# competed for the same small free-tier quota.


def test_provider_chain_uses_both_when_both_are_configured(monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "groq,gemini")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")
    assert n.configured_providers() == ("groq", "gemini")


def test_provider_chain_skips_a_provider_with_no_key(monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "groq,gemini")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")
    assert n.configured_providers() == ("gemini",)


def test_provider_chain_honours_the_configured_order(monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "gemini,groq")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")
    assert n.configured_providers() == ("gemini", "groq")


def test_ollama_needs_no_key_to_be_considered(monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "ollama")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "")
    assert n.configured_providers() == ("ollama",)


def test_an_unknown_provider_name_is_ignored(monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "hal9000,gemini")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")
    assert n.configured_providers() == ("gemini",)


async def test_narrate_uses_the_first_provider_that_works(dev_brief, monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "groq,gemini")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")
    tried = []

    async def fake(provider, prompt):
        tried.append(provider)
        return n.NarrativeDraft(summary="Groq wrote this.")

    monkeypatch.setattr(n, "_call_provider", fake)
    draft = await n.narrate_llm(dev_brief, e.escalate(dev_brief))
    assert tried == ["groq"]
    assert draft.summary == "Groq wrote this."


async def test_narrate_falls_through_to_gemini_when_groq_fails(dev_brief, monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "groq,gemini")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")
    tried = []

    async def fake(provider, prompt):
        tried.append(provider)
        if provider == "groq":
            raise RuntimeError("429 rate limit")
        return n.NarrativeDraft(summary="Gemini wrote this.")

    monkeypatch.setattr(n, "_call_provider", fake)
    draft = await n.narrate_llm(dev_brief, e.escalate(dev_brief))
    assert tried == ["groq", "gemini"]
    assert draft.summary == "Gemini wrote this."


async def test_an_empty_reply_moves_on_to_the_next_provider(dev_brief, monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "groq,gemini")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")
    tried = []

    async def fake(provider, prompt):
        tried.append(provider)
        if provider == "groq":
            return n.NarrativeDraft(summary="   ")   # schema-valid, useless
        return n.NarrativeDraft(summary="Gemini wrote this.")

    monkeypatch.setattr(n, "_call_provider", fake)
    draft = await n.narrate_llm(dev_brief, e.escalate(dev_brief))
    assert tried == ["groq", "gemini"]
    assert draft.summary == "Gemini wrote this."


async def test_narrate_raises_when_every_provider_fails(dev_brief, monkeypatch):
    # analyze() relies on this raising so it can render the template instead.
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "groq,gemini")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "gem_test")

    async def fake(provider, prompt):
        raise RuntimeError(f"{provider} is down")

    monkeypatch.setattr(n, "_call_provider", fake)
    with pytest.raises(Exception):
        await n.narrate_llm(dev_brief, e.escalate(dev_brief))


async def test_narrate_raises_when_nothing_is_configured(dev_brief, monkeypatch):
    monkeypatch.setattr(n.settings, "REASONING_PROVIDERS", "groq,gemini")
    monkeypatch.setattr(n.settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(n.settings, "GEMINI_API_KEY", "")
    with pytest.raises(Exception):
        await n.narrate_llm(dev_brief, e.escalate(dev_brief))


def test_prompt_demands_plain_language_and_questions(dev_brief):
    prompt = n.build_prompt(dev_brief, e.escalate(dev_brief))
    lowered = prompt.lower()
    assert "plain" in lowered
    assert "three" in lowered or "3" in prompt          # how many questions
    assert "jargon" in lowered


def test_fallback_explains_a_result_in_plain_words(dev_brief):
    draft = n.render_fallback(dev_brief, e.escalate(dev_brief))
    joined = " ".join(draft.key_findings)
    assert "infection-fighting" in joined


def test_fallback_always_offers_questions_when_something_is_abnormal(dev_brief):
    draft = n.render_fallback(dev_brief, e.escalate(dev_brief))
    assert len(draft.doctor_questions) >= 3
