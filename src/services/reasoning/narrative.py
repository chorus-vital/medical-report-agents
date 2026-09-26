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

from config.settings import settings
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
    interval = (f" (usual range {abnormality.range_text})"
                if abnormality.range_text else "")
    gloss = f" — {abnormality.plain_meaning}" if abnormality.plain_meaning else ""
    return (f"{abnormality.test_name} is {abnormality.value}{unit}, "
            f"{abnormality.direction} the usual range{interval}{gloss}")


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
        questions.append(f"What could explain the results outside the usual "
                         f"range ({names})?")
        questions.append("Do any of these results need to be repeated, and if "
                         "so, when?")
        questions.append("Do these results change anything about my current care?")
        questions.append("Is there anything I should watch out for before my "
                         "next appointment?")

    tips: List[str] = []
    if abnormal:
        tips.append("Bring this report with you to your appointment.")
        tips.append("Write down any recent symptoms and when they started.")

    return NarrativeDraft(
        summary=" ".join(parts),
        key_findings=findings,
        doctor_questions=questions,
        lifestyle_tips=tips,
    )


_SYSTEM_RULES = """\
You are writing a plain-language explanation of one lab report for the patient
who took it. Follow these rules exactly.

1. Use only the facts in the BRIEF below. Never state a number that does not
   appear there.
2. Never diagnose. Do not name a disease, condition, or cause. Do not write
   "anaemia", "infection", "deficiency", or any similar term. Do not write
   "this indicates", "this suggests", "this means", or "you have".
3. Describe what a value is and whether it is inside or outside its range.
   Nothing more.
4. Never describe a result the brief does not list as outside its range, and
   say nothing at all about whether a NOT EVALUATED result is normal.
5. If the brief has lab notes, you may quote one word-for-word inside quotation
   marks. Never paraphrase a lab note.
6. Write in plain everyday English, the way you would explain it to a friend
   with no medical background. Short sentences. Say "white blood cells, which
   fight infection" rather than "leukocytes". The brief gives you a "what it
   measures" line for each result — use those words. Avoid jargon; when you must
   use a lab term, gloss it immediately in ordinary words.
7. The summary is for a worried person skimming on a phone. Three or four short
   sentences. Say which results are outside the range and roughly what those
   results are about — do not list every number there; the findings list does
   that.
8. Always give three to five doctor_questions. They are the questions this
   patient should actually ask, specific to what came back abnormal, phrased in
   their own voice.
9. lifestyle_tips are practical, non-medical preparation steps. Never suggest a
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


# Which setting must hold a key before a provider is worth trying. Ollama runs
# locally and needs none.
_PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": None,
}


def configured_providers() -> tuple:
    """
    The fallback chain, in order, with unconfigured providers dropped.

    Names that are not providers are ignored rather than raising: a typo in
    REASONING_PROVIDERS should cost one provider, not the whole analysis.
    """
    chain = []
    for name in (settings.REASONING_PROVIDERS or "").split(","):
        name = name.strip().lower()
        if name not in _PROVIDER_KEYS:
            if name:
                logger.warning("Ignoring unknown reasoning provider %r", name)
            continue
        key_setting = _PROVIDER_KEYS[name]
        if key_setting and not getattr(settings, key_setting, ""):
            logger.debug("Skipping %s: %s not set", name, key_setting)
            continue
        if name not in chain:
            chain.append(name)
    return tuple(chain)


async def _narrate_gemini(prompt: str) -> NarrativeDraft:
    """Gemini's native structured output, which constrains decoding to the schema."""
    from google.genai import types as genai_types

    from src.services.extractor import _with_retries
    from src.services.llm_factory import get_vision_model

    client = get_vision_model()
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

    response = await _with_retries(call, "Agent 3 narrative (gemini)")

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, NarrativeDraft):
        return parsed
    if isinstance(parsed, dict):
        return NarrativeDraft.model_validate(parsed)
    return NarrativeDraft.model_validate_json(getattr(response, "text", "") or "{}")


async def _narrate_langchain(provider: str, prompt: str) -> NarrativeDraft:
    """
    Groq and Ollama through LangChain's structured-output binding.

    Neither has Gemini's ``response_schema``, so the shape is enforced by tool
    calling or JSON mode depending on the model. That is a weaker guarantee, but
    the verifier does not trust any of them anyway — a malformed reply raises
    here and the chain moves on.
    """
    from src.services.extractor import _with_retries
    from src.services.llm_factory import get_chat_model

    model = get_chat_model(temperature=0.2, provider=provider)
    structured = model.with_structured_output(NarrativeDraft)

    async def call():
        return await structured.ainvoke(prompt)

    result = await _with_retries(call, f"Agent 3 narrative ({provider})")

    if isinstance(result, NarrativeDraft):
        return result
    if isinstance(result, dict):
        return NarrativeDraft.model_validate(result)
    raise ValueError(f"{provider} returned {type(result).__name__}, not a narrative")


async def _call_provider(provider: str, prompt: str) -> NarrativeDraft:
    """Dispatch one provider. Raises on any failure, so the chain can move on."""
    if provider == "gemini":
        return await _narrate_gemini(prompt)
    return await _narrate_langchain(provider, prompt)


async def narrate_llm(brief: ClinicalBrief, escalation: Escalation,
                      violations: tuple = ()) -> NarrativeDraft:
    """
    Ask each configured provider in turn for the narrative. Raises if none works.

    Raising is the contract ``analyze`` depends on: it renders the verified
    template instead, so an outage degrades the wording rather than emptying the
    analysis.
    """
    providers = configured_providers()
    if not providers:
        raise RuntimeError(
            "No reasoning provider is configured. Set GROQ_API_KEY (free at "
            "https://console.groq.com) or GEMINI_API_KEY, or point "
            "REASONING_PROVIDERS at a running Ollama."
        )

    prompt = build_prompt(brief, escalation, violations)
    last: Exception | None = None

    for provider in providers:
        try:
            draft = await _call_provider(provider, prompt)
        except Exception as exc:
            last = exc
            logger.warning("Agent 3 narrative via %s failed (%s) — trying next",
                           provider, str(exc)[:200])
            continue

        if not is_usable(draft):
            last = ValueError(f"{provider} returned an empty narrative")
            logger.warning("Agent 3 narrative via %s was empty — trying next",
                           provider)
            continue

        logger.info("Agent 3 narrative written by %s", provider)
        return draft

    raise last or RuntimeError("No reasoning provider produced a narrative")
