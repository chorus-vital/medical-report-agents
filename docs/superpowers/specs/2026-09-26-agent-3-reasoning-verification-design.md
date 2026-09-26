# Agent 3 — Reasoning & Verification

**Date:** 2026-09-26
**Status:** Approved design, pending implementation plan
**Branch:** `SF/final-agent`

## Purpose

Agent 3 is the last node in the pipeline. It turns Agent 2's coded, flagged
lab rows into something a patient can act on: a plain-language summary, the
findings that matter, questions worth asking a doctor, and relevant lifestyle
notes — plus an urgency level and an honest confidence score.

The problem it has to solve is not fluency. An LLM will write a readable
summary of a blood report unprompted. The problem is that a readable summary
of a blood report is exactly the kind of text where a fabricated number or an
invented test reads as authoritative, and where a casual "this suggests"
turns an information tool into an unlicensed diagnosis.

This carries forward the principle the terminology module already states:
*a wrong LOINC code is worse than no code*. The same holds for narrative. A
claim that cannot be traced to a row is worse than silence.

### Success criteria

1. Every number and every analyte named in the output traces to a row Agent 2
   produced. Violations are detected in code, not by prompt adherence.
2. No candidate diagnosis is ever named.
3. The pipeline produces a usable analysis when the LLM is unavailable.
4. `confidence_score` is computed from observable signals and is never
   self-reported by the model.
5. The whole reasoning layer is testable with no API key.

### Non-goals

- Naming conditions, even likely ones. See "Reasoning scope" below.
- Treatment, dosage, or medication advice.
- Trend analysis across multiple reports. There is one report per run.
- Replacing Agent 2's flags with a second opinion on normality.

## Decisions taken

Four decisions shaped this design. They were chosen deliberately and the
alternatives were live options.

**Verification is deterministic, not a second LLM pass.** A self-critique
pass shares the blind spots of the pass that wrote the text, and it cannot be
tested without burning API calls. A code verifier that checks claims against
a frozen fact set gives a floor that does not depend on model behaviour.

**Reasoning stops at facts and urgency; it does not name conditions.** The
test report shows low MCV with a Mentzer index of 14.8, which is a textbook
iron-deficiency-versus-thalassaemia pattern. Agent 3 will not say so. It
reports the values, groups them, sets an urgency level, and surfaces the
lab's own note verbatim. Naming conditions would make the repo responsible
for defending clinical inferences to patients reading them alone.

**Agent 2's AMBER band is fixed first.** The current margin is 10% of the
interval *width*, so a range of 20–500 tolerates a value of 0 as "borderline".
Escalation derives from these flags, so the fix lands before Agent 3 is tuned.

**An LLM outage degrades the output, it does not empty it.** Agent 1 already
falls back to a regex extractor and marks the run degraded. Agent 3 follows
that precedent with a templated renderer over the same verified brief.

## Architecture

```
lab_results ──► build_brief() ──► ClinicalBrief ──► escalate()
   (Agent 2)      no LLM            frozen facts       no LLM
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                    narrate_llm(brief)        render_fallback(brief)
                     structured Gemini          no LLM, templated
                            │                         │
                            ▼                         │
                      verify(draft, brief)            │
                         no LLM                       │
                            │                         │
              ┌─────────────┼──────────────┐          │
              ▼             ▼              ▼          │
           clean      drop bullets    retry once ──────┤
              │             │              │          │
              └─────────────┴──────────────┴──────────►│
                                                       ▼
                                             score_confidence()
                                                  no LLM
                                                       │
                                                       ▼
                                              ReasoningResult
```

Four of the five modules never touch the network.

### Module layout

Agent 3 is a package rather than a single module. `src/services/extractor.py`
is around a thousand lines and is hard to work in; splitting Agent 3 along its
natural seams avoids repeating that. This is a deliberate deviation from the
flat `services/*.py` pattern used by Agents 1 and 2.

| File | Responsibility | LLM |
|---|---|---|
| `reasoning/__init__.py` | `analyze()` — the only public entry point | orchestrates |
| `reasoning/brief.py` | `ClinicalBrief` and its builder | no |
| `reasoning/escalation.py` | urgency rule pack | no |
| `reasoning/narrative.py` | prompt, Gemini call, fallback renderer | yes |
| `reasoning/verify.py` | claim verifier | no |

`nodes.py` imports `analyze` and nothing else.

## Components

### 1. The brief

`build_brief()` is the only thing that reads Agent 2's rows. It produces a
frozen fact set; the LLM never sees raw rows. This matters because the brief
*defines* what the verifier will accept — the allowed facts are explicit
rather than implied.

```python
@dataclass(frozen=True)
class Abnormality:
    test_name: str            # standard_name when coded, else printed name
    value: str
    unit: str | None
    range_text: str
    flag: str                 # "AMBER" | "RED"
    direction: str            # "below" | "above"
    panel: str | None

@dataclass(frozen=True)
class ClinicalBrief:
    total_rows: int
    coded_rows: int
    unknown_rows: int
    flag_counts: dict[str, int]           # GREEN/AMBER/RED/UNKNOWN
    abnormalities: tuple[Abnormality, ...]  # RED first, then AMBER
    normal_panels: tuple[str, ...]        # panels with no abnormality
    lab_notes: tuple[str, ...]            # report_notes, verbatim
    patient_age: str | None
    patient_sex: str | None
    extraction_degraded: bool
    allowed_numbers: frozenset[str]       # for the verifier
    allowed_analytes: frozenset[str]      # for the verifier
```

`allowed_numbers` holds every observed value, every range bound, every count
the brief exposes, and the patient's age, in normalised string form (`"5.0"`
and `"5"` both normalise to `"5"`). `allowed_analytes` holds the standard and
printed names of rows actually present in this report.

### 2. Escalation

`escalate(brief) -> Escalation` applies a small rule pack. Four levels:

| Level | Meaning |
|---|---|
| `routine` | nothing outside range |
| `discuss_at_next_visit` | AMBER only |
| `see_doctor_promptly` | any RED, or a moderate threshold rule |
| `seek_care_now` | a critical-value rule fired |

```python
@dataclass(frozen=True)
class FiredRule:
    rule_id: str        # "anc_moderate"
    test_name: str      # the row that triggered it
    detail: str         # "ANC 1035 below 1500"

@dataclass(frozen=True)
class Escalation:
    level: str
    reasons: tuple[FiredRule, ...]
```

The rule pack is deliberately short and keyed on published critical values:

| Rule id | Condition | Level |
|---|---|---|
| `anc_critical` | ANC < 500 | `seek_care_now` |
| `platelet_critical` | platelets < 20,000 | `seek_care_now` |
| `hb_critical` | haemoglobin < 7 | `seek_care_now` |
| `potassium_critical` | K < 2.5 or > 6.5 | `seek_care_now` |
| `glucose_critical` | glucose < 50 or > 500 | `seek_care_now` |
| `anc_moderate` | ANC < 1500 | `see_doctor_promptly` |
| `platelet_low` | platelets < 50,000 | `see_doctor_promptly` |
| `hb_low` | haemoglobin < 10 | `see_doctor_promptly` |
| `any_red` | any RED flag | `see_doctor_promptly` |
| `any_amber` | any AMBER, no RED | `discuss_at_next_visit` |

Rules match on `ontology_key`, not on printed names, so they only fire on rows
Agent 2 coded confidently. An uncoded row cannot trigger escalation — it
contributes to lowered confidence instead.

The highest level wins, and every fired rule is retained so the level can be
explained. On the test report `anc_moderate` and `any_red` both fire, giving
`see_doctor_promptly`.

The two aggregate rules (`any_red`, `any_amber`) are not tied to one row. They
cite the most severe abnormality — the first entry in `brief.abnormalities` —
so that `FiredRule.test_name` is always populated.

**This table is the part of the design most worth scrutiny.** It is clinical
content living in a repo. It is small and it is sourced, but it is a
maintenance obligation, and a wrong threshold here is a safety bug rather than
a correctness bug.

### 3. Narrative

One structured-output Gemini call. The response schema carries no confidence
field — the model is never asked how sure it is.

```python
class NarrativeDraft(BaseModel):
    summary: str
    key_findings: list[str]
    doctor_questions: list[str]
    lifestyle_tips: list[str]
```

The prompt supplies the rendered brief and constrains the model to: plain
language at roughly an eighth-grade reading level; no condition names; no
diagnosis verbs; cite only values present in the brief; never characterise a
row the brief marks UNKNOWN; reproduce lab notes verbatim if quoting them.

`render_fallback(brief)` produces the same four fields from templates over the
same brief, without an LLM. It is flatter prose, and it is used on any LLM
failure, on repeated verification failure, and in every offline test. It is
not a stub: it is the guaranteed floor of the feature.

### 4. Verifier

`verify(draft, brief) -> VerificationReport`. Four checks, no LLM:

1. **Numbers.** Every numeric token in the text must normalise into
   `allowed_numbers`. Catches fabricated values.
2. **Analytes.** The full ontology's names and aliases are scanned against the
   text; any analyte that resolves but is *not* in `allowed_analytes` is a
   violation. Catches "your platelets are normal" when platelets were never
   tested.
3. **Diagnosis language.** A blocklist of verbs (`you have`, `indicates`,
   `suggests`, `caused by`, `diagnos*`) and condition vocabulary.
4. **UNKNOWN rows.** A row Agent 2 could not evaluate must not be described as
   normal or abnormal.

Handling: a failing bullet is dropped from its list. A failing `summary`
triggers one LLM retry with the violations named. Still failing, the whole
draft is discarded for `render_fallback`.

**Stated limit.** This bounds fabrication, it does not eliminate it. A claim
built from real numbers and no recognised analyte name passes all four checks.
The verifier makes invented *facts* detectable; it cannot make invented
*reasoning* detectable. That is a large part of why reasoning scope stops at
facts and urgency.

### 5. Confidence

`score_confidence(brief, verification) -> float`, in 0.0–1.0, deterministic:

```
score = 1.0
      × (0.5 if extraction_degraded else 1.0)
      × (0.6 + 0.4 × coded_rows / total_rows)
      × (1.0 − 0.5 × unknown_rows / total_rows)
      × (1.0 clean | 0.7 claims dropped | 0.4 fell back to template)
```

Clamped to [0, 1] and rounded to two places. Each factor is independently
testable, and a degraded extraction can never present as a confident analysis.

`total_rows == 0` returns 0.0 before the formula runs, so the ratio terms never
divide by zero.

## Data flow and wiring

`reason_and_verify_node` consumes `lab_results`, `patient_info`,
`report_notes` and `extraction_degraded`, and produces `summary`,
`key_findings`, `doctor_questions`, `lifestyle_tips`, `confidence_score`
(all already declared in `PipelineState`) plus three new fields:
`escalation_level`, `escalation_reasons`, `reasoning_degraded`.

`ReportAnalysis` gains the same three. `_build_response` gains a `reasoning`
block beside the existing `grounding` block, following that convention:

```json
"reasoning": {
  "escalation_level": "see_doctor_promptly",
  "escalation_reasons": [
    {"rule_id": "anc_moderate", "test_name": "Absolute Neutrophil Count",
     "detail": "ANC 1035 below 1500"}
  ],
  "degraded": false,
  "confidence_score": 0.82
}
```

`index.html` renders the summary, the three lists, and an escalation banner
coloured by level, with the existing disclaimer kept in place.

### Error handling

| Failure | Behaviour |
|---|---|
| No lab rows | Empty analysis, `routine`, confidence 0.0, no LLM call |
| LLM unavailable | `render_fallback`, `reasoning_degraded = True` |
| Draft fails verification | Drop bullets; retry summary once; then fallback |
| Escalation rule raises | Log, treat as not fired, continue |
| Node raises | Caught in `nodes.py`, error appended to `errors`, pipeline completes |

The node never breaks the pipeline. This matches `extract_node`'s existing
contract.

## Testing

Everything below runs with no API key.

**Unit.** Per module. `brief.py`: counts, ordering, allowed-set construction.
`escalation.py`: each rule at, above and below its threshold; highest-level-wins;
uncoded rows cannot fire a rule. `verify.py`: deliberately poisoned drafts —
an invented value, an untested analyte, "this indicates anaemia", a claim about
an UNKNOWN row — each must be caught. `narrative.py`: the fallback renderer's
output against a known brief.

**Fixture.** Agent 2's real output for `Dev chavan Report.pdf` is committed as
`tests/fixtures/dev_chavan_agent2.json` and drives the integration tests. The
report is a genuinely good case: 22 rows, leukopenia with neutropenia across
three white-cell subtypes, a low MCV, and a doctor's note on the page.

**Integration, offline.** Fixture → brief → escalation → fallback → verify →
confidence, asserting `see_doctor_promptly`, the specific fired rules, and that
the fallback narrative passes its own verifier. A narrative that cannot pass
verification is a bug in the narrative, including when we wrote it ourselves.

**Live.** One end-to-end test on the real PDF behind the existing `live` marker
in `pytest.ini`, deselected by default.

## Implementation order

1. AMBER band fix in `evaluate_flag` — `margin = |breached bound| × 0.10`.
   Existing flag tests pass unchanged under this rule; verify that first.
2. Capture the fixture from a real run.
3. `brief.py`, then `escalation.py`, then `verify.py`, then `narrative.py` —
   each test-first, each independently useful.
4. `analyze()` orchestration.
5. Wire `nodes.py`, schemas, `_build_response`, `index.html`.
6. Integration tests, then the live test.

Steps 3 and 5 are the natural review points.

## Open risk

The escalation rule pack is clinical content maintained by people who are not
clinicians. The mitigation is that it is small, every threshold is a published
critical value, every fired rule is traceable to the row that fired it, and the
output never states a cause — only that a number is outside a range and how
soon to act on it. It is worth a clinician's eye before this ships to real
users.
