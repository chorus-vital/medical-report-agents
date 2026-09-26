# Medical Report Analyzer

Upload a lab report, get it read back to you in plain English — with every number and every test name checked, in code, against the rows the document actually contained.

The hard part is not fluency. A language model will happily write a readable summary of a blood report. The problem is that a readable summary of a blood report is exactly the kind of text where a fabricated number reads as authoritative, and where a casual *"this suggests"* turns an information tool into an unlicensed diagnosis.

So the architecture inverts the usual arrangement: **the model writes the prose, but it does not decide what is true.** A deterministic brief is built in code and is the only thing the model sees; a deterministic verifier then checks every claim in the reply against that brief and drops what it cannot trace.

The governing principle, carried through every layer: **a wrong code is worse than no code, and a claim that cannot be traced to a row is worse than silence.**

---

## The pipeline

Three agents, orchestrated as a LangGraph `StateGraph`.

```
 PDF / image / docx / text
            │
            ▼
┌───────────────────────────┐
│  AGENT 1  Ingestion       │  pdfplumber + pypdfium2, Gemini structured
│           & Extraction    │  extraction, Gemini Vision for scans,
│                           │  regex fallback when the LLM is unreachable
└───────────────────────────┘
            │  extracted_items, patient_info
            ▼
┌───────────────────────────┐
│  AGENT 2  Terminology     │  LOINC matching against a local ontology,
│           & Flagging      │  reference-range evaluation → GREEN / AMBER /
│                           │  RED / UNKNOWN.  No LLM at all.
└───────────────────────────┘
            │  lab_results
            ▼
┌───────────────────────────┐
│  AGENT 3  Reasoning       │  deterministic brief → LLM narrative →
│           & Verification  │  deterministic verifier → confidence score
└───────────────────────────┘
            │
            ▼
     summary, key findings, questions for your doctor,
     urgency level, confidence
```

### How Agent 3 keeps itself honest

```
lab_results ──► build_brief()  ──►  ClinicalBrief  ──► escalate()
                  no LLM            frozen facts        no LLM
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                    narrate_llm(brief)        render_fallback(brief)
                    Groq, then Gemini           no LLM, templated
                            │                         │
                            ▼                         │
                      verify(draft, brief)            │
                          no LLM                      │
                            │                         │
              ┌─────────────┼──────────────┐          │
              ▼             ▼              ▼          │
           clean      drop bullets    retry once ─────┤
              │             │              │          │
              └─────────────┴──────────────┴─────────►│
                                                      ▼
                                            score_confidence()
```

Four of Agent 3's five modules never touch the network. The verifier runs four checks against the brief:

| Check | Catches |
|---|---|
| **Numbers** | A value that appears nowhere in the report (including spelled-out decimals) |
| **Analytes** | *"Your platelets are fine"* when platelets were never tested |
| **Diagnosis language** | Naming a condition, or attributing one to the patient |
| **Un-evaluated rows** | Calling a result "normal" that Agent 2 could not evaluate |

**A stated limit:** this bounds fabricated *facts*; it cannot bound unsound *reasoning*. A claim built from real numbers that names no recognised analyte passes all four checks. That is a large part of why the product deliberately stops at facts and urgency, and never names a condition.

---

## Prerequisites

- **Python 3.10+** (developed on 3.13)
- **Node 20.19+ or 22.12+** — only needed for the React frontend
- A **Groq** API key (free, no card) — <https://console.groq.com>
- A **Gemini** API key (free tier) — <https://aistudio.google.com>

Gemini is required for extraction: it is the only one of the three providers with a usable free vision API, which scanned reports need. Groq handles Agent 3's narrative, which keeps the two agents off the same quota.

---

## Setup

### 1. Clone and enter

```bash
git clone https://github.com/chorus-vital/medical-report-agents.git
cd medical-report-agents
```

### 2. Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure keys

`.env` is gitignored, so create it in the project root:

```ini
# Extraction (Agent 1) — needs Gemini for vision
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash

# Reasoning (Agent 3) — ordered fallback chain, tried left to right.
# A provider with no key is skipped. If all fail, Agent 3 still returns a
# verified templated analysis rather than nothing.
REASONING_PROVIDERS=groq,gemini
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

> **Model ids go stale.** Groq retires models; `llama-3.3-70b-versatile` no longer exists. If you see a 404 from Groq, list what your key can reach:
> ```bash
> python -c "from groq import Groq; from config.settings import settings; print([m.id for m in Groq(api_key=settings.GROQ_API_KEY).models.list().data])"
> ```

### 4. Build the frontend (optional)

Only needed for the React reader at `/app`. The original webview at `/` works without it.

```bash
cd frontend
npm install
npm run build      # outputs to src/static/app/, served by FastAPI
cd ..
```

---

## Running it

```bash
python main.py
```

| URL | What it is |
|---|---|
| <http://localhost:8000/app> | React + GSAP reader — scroll-driven, with a 3D cell view |
| <http://localhost:8000/> | Original testing webview — dense table, raw JSON |
| <http://localhost:8000/docs> | OpenAPI / Swagger |
| <http://localhost:8000/api/health/llm> | Live provider connectivity check |

### Frontend development

Hot reload, with `/api` proxied to the backend. **Both** servers must be running.

```bash
python main.py                 # terminal 1
cd frontend && npm run dev     # terminal 2 → http://localhost:5173
```

---

## Testing

```bash
# 269 offline tests. No API key needed, no quota consumed, ~2s.
python -m pytest -q

# Hits the real LLM APIs. Deselected by default.
python -m pytest -m live -q
```

The reasoning tests run against `tests/fixtures/dev_chavan_agent2.json` — real Agent 2 output captured from a real report — so the whole layer is testable offline. Re-capture it after an ontology or extractor change:

```bash
python scripts/capture_fixture.py "path/to/report.pdf" tests/fixtures/dev_chavan_agent2.json
```

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/reports/analyze` | Upload a file, run the full pipeline |
| `POST` | `/api/reports/analyze-text` | Same, from pasted text |
| `GET` | `/api/reports/samples` | List bundled sample reports |
| `POST` | `/api/reports/analyze-sample/{filename}` | Run a bundled sample |
| `GET` | `/api/health` | Service health |
| `GET` | `/api/health/llm` | Live provider reachability |

### Response shape

```json
{
  "status": "success",
  "patient_info": { "name": "...", "age": "23 Year(s)", "sex": "Male" },
  "lab_results": [
    {
      "test_name": "Total White Blood Cell Count (TC)",
      "standard_name": "Total WBC Count",
      "loinc_code": "6690-2",
      "observed_value": "2130",
      "unit": "cells/mm³",
      "reference_range": "4000 - 10000",
      "flag": "RED",
      "explanation": "The number of infection-fighting white blood cells.",
      "match_confidence": 100.0,
      "range_source": "report"
    }
  ],
  "extraction": { "method": "gemini:...", "degraded": false, "warnings": [] },
  "grounding": { "loinc_coded_count": 21, "loinc_coded_of": 25, "flag_summary": {} },
  "summary": "Several of your blood counts are lower than the usual range...",
  "key_findings": ["..."],
  "doctor_questions": ["..."],
  "lifestyle_tips": ["..."],
  "reasoning": {
    "escalation_level": "see_doctor_promptly",
    "escalation_reasons": [
      { "rule_id": "anc_moderate", "test_name": "Absolute Neutrophil Count",
        "detail": "Absolute Neutrophil Count 1035 is below 1500 /mm3" }
    ],
    "degraded": false,
    "confidence_score": 0.82,
    "dropped_claims": 0
  }
}
```

Two fields worth understanding:

- **`extraction.degraded`** — Agent 1 fell back to the regex parser, so the row list may be incomplete.
- **`reasoning.degraded`** — Agent 3 wrote from its template because no LLM was reachable. **These are different failures** and are reported separately; a degraded extraction does not mean the wording is templated.

---

## Escalation levels

Derived in code from published critical values, never from the model. Every level is traceable to the rule and row that produced it. Units are converted from each row's **declared** unit, and a rule refuses to fire on an unrecognised one.

| Level | Meaning |
|---|---|
| `no_data` | No results could be read |
| `routine` | Nothing outside its reference range |
| `discuss_at_next_visit` | Borderline results only |
| `see_doctor_promptly` | Any result outside range, or a moderate threshold rule |
| `seek_care_now` | A critical value fired |

---

## Project structure

```
medical-report-agents/
├── src/
│   ├── api/                      FastAPI app and routes
│   ├── graph/                    LangGraph nodes and pipeline
│   ├── schemas/                  Pydantic models, pipeline state
│   ├── services/
│   │   ├── extractor.py          Agent 1
│   │   ├── terminology.py        Agent 2
│   │   ├── pdf_text.py           Dual-engine PDF text + mojibake repair
│   │   ├── llm_factory.py        Gemini / Groq / Ollama
│   │   └── reasoning/            Agent 3
│   │       ├── __init__.py       analyze(), score_confidence()
│   │       ├── brief.py          the frozen fact set
│   │       ├── escalation.py     urgency rules
│   │       ├── narrative.py      provider chain + templated fallback
│   │       └── verify.py         the four checks
│   └── static/
│       ├── index.html            original webview
│       └── app/                  built React bundle (gitignored)
├── frontend/                     Vite + React 19 + GSAP + three.js
├── data/lab_ontology.json        55 analytes, LOINC codes, ranges
├── docs/superpowers/             design specs and implementation plans
├── scripts/capture_fixture.py    regenerate the test fixture
└── tests/                        269 offline + 2 live
```

---

## Known limits

1. **The escalation thresholds are clinical content maintained by non-clinicians.** Every value is a published critical value and every fired rule is traceable, but **nobody qualified has reviewed the numbers.** This warrants a clinician's eye before real patients see it.
2. **The verifier bounds invented facts, not invented reasoning** — stated plainly in `verify.py`'s own docstring rather than glossed over.
3. **No persistence.** `SQLITE_DB_PATH` is configured but nothing writes to it; every analysis is lost once the response is sent.
4. **No SNOMED CT, no FHIR export.** LOINC only.
5. **The frontend has no automated tests.**
6. **Gemini's free tier is small.** Heavy testing exhausts it, after which Agent 1 falls back to the regex parser — visible as a degraded-extraction warning and a lower confidence score.

---

## Safety

- No condition is ever named. The output describes what a value is, whether it sits inside its interval, and how soon to act.
- The 3D cell view is **anatomical, never pathological** — it shows what a cell is and how many there are, never what a low count does to you. A picture cannot be checked the way the verifier checks prose.
- A lab's own printed notes are reproduced **verbatim and attributed**, never paraphrased.
- Every response carries a disclaimer. This is not a diagnosis, and it is not a substitute for a clinician.

---

## Licence

Not yet chosen. Until a `LICENSE` file is added, default copyright applies and
the code carries no grant of use — worth settling before this goes public.
