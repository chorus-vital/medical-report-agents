# Simplified Architecture Document — Agents Backend
## `medical-report-agents` (Localhost & Free-Tier Edition)

**Version:** 2.0 (Simplified for Localhost & Free APIs)  
**Date:** August 2026  
**Stack:** Python 3.11+ · FastAPI · LangGraph · SQLite · Free AI APIs (Google Gemini Flash / Local Ollama / Groq)

---

## 1. Design Philosophy

The goal is a **highly useful, zero-cost, and easy-to-run system** that runs completely on localhost with **no paid APIs**, **no heavy cloud infrastructure**, and **no complex multi-container requirements**.

| Aspect | Original Over-Engineered Architecture | Simplified Localhost Architecture |
|---|---|---|
| **LLM Provider** | Paid Anthropic Claude API (Sonnet/Opus) | **100% Free**: Google Gemini 2.0/1.5 Flash API (free tier) + Local Ollama (`llama3.2` / `qwen2.5`) + Groq Free Tier |
| **Database** | Heavy PostgreSQL + pgvector + connection pools | **SQLite** (`medical_reports.db`) — zero setup, single-file storage |
| **Terminology Server** | Heavy Java-based Snowstorm FHIR container (4-8GB RAM) | **Local Embedded Lab Dictionary** (JSON/SQLite) covering 100+ standard lab tests (CBC, Lipid, LFT, KFT, Thyroid, etc.) with LOINC & reference ranges |
| **Document Parsing** | Heavy IBM Docling + PyTorch PubMedBERT NER models | **Gemini Vision (Multimodal)** / `pdfplumber` / `pytesseract` / regex-assisted structured extractor |
| **State Persistence** | Postgres Checkpointer (`AsyncPostgresSaver`) | **MemorySaver** / SQLite Checkpointer |
| **PDF Generation** | WeasyPrint (requires C libraries: Cairo/Pango) | **ReportLab** / Frontend Browser Print Layout (`window.print()`) |
| **Auth & Security** | JWT RBAC + Firebase Auth + S3 signed URLs | **Local-first / Single User mode** (zero auth friction for development) |
| **Agent Complexity** | 7 fragmented micro-agents with routing loops | **3 Streamlined High-Efficiency Agent Nodes** |

---

## 2. High-Level System Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI APPLICATION (http://localhost:8000)             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                            API Endpoints                               │  │
│  │  POST /api/reports/analyze (SSE / Stream)    GET /api/reports/history  │  │
│  │  GET  /api/reports/{id}                      GET /api/trends           │  │
│  │  POST /api/reports/upload                    GET /api/health           │  │
│  └───────────────────────────────────┬────────────────────────────────────┘  │
│                                      │                                        │
│  ┌───────────────────────────────────▼────────────────────────────────────┐  │
│  │                 STREAMLINED LANGGRAPH PIPELINE                         │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │ 1. Ingestion & Extraction Agent                                  │  │  │
│  │  │    • Input: PDF / Image / Text                                   │  │  │
│  │  │    • Engine: Gemini Flash Vision / pdfplumber                    │  │  │
│  │  │    • Output: Structured Lab Items (Test, Value, Unit, Raw Range) │  │  │
│  │  └────────────────────────────────┬─────────────────────────────────┘  │  │
│  │                                   │                                    │  │
│  │  ┌────────────────────────────────▼─────────────────────────────────┐  │  │
│  │  │ 2. Terminology & Flagging Agent                                  │  │  │
│  │  │    • Engine: Local Lab Dictionary (`data/lab_ontology.json`)     │  │  │
│  │  │    • Action: Maps test to LOINC, checks age/sex normal ranges    │  │  │
│  │  │    • Output: Green (Normal) / Amber (Borderline) / Red (Alert)   │  │  │
│  │  └────────────────────────────────┬─────────────────────────────────┘  │  │
│  │                                   │                                    │  │
│  │  ┌────────────────────────────────▼─────────────────────────────────┐  │  │
│  │  │ 3. Reasoning & Verification Agent                                │  │  │
│  │  │    • Engine: Gemini Flash / Local Ollama (Structured Prompt)     │  │  │
│  │  │    • Action: Plain-language summary + Questions for doctor       │  │  │
│  │  │    • Verifier Check: Cross-checks claims vs extracted values     │  │  │
│  │  │    • Output: Verified explanation + safety disclaimer            │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────┬────────────────────────────────────┘  │
│                                      │                                        │
│  ┌───────────────────────────────────▼────────────────────────────────────┐  │
│  │                     LOCAL DATA & STORAGE                               │  │
│  │  • SQLite Database: `data/medical_reports.db` (Reports & Lab Results)  │  │
│  │  • Uploads Folder:  `data/uploads/` (Original PDFs/Images)             │  │
│  │  • Ontology:        `data/lab_ontology.json` (LOINC + Reference Ranges)│  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Simplified Repository Structure

```
medical-report-agents/
├── .env.example                    # GEMINI_API_KEY, OLLAMA_BASE_URL, MODEL_PROVIDER
├── README.md                       # 1-command startup instructions
├── requirements.txt                # Lightweight, fast-installing dependencies
├── main.py                         # Single entrypoint: python main.py
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Pydantic Settings (API keys, SQLite path, provider)
│
├── data/
│   ├── lab_ontology.json           # Embedded 100+ lab tests (LOINC, normal ranges, units)
│   ├── medical_reports.db          # Auto-created SQLite database
│   ├── samples/                    # Sample test reports for instant testing
│   └── uploads/                    # Local storage for uploaded files
│
├── src/
│   ├── __init__.py
│   │
│   ├── schemas/                    # Pydantic Data Models
│   │   ├── report.py               # LabResult, ReportAnalysis, FlagEnum (GREEN/AMBER/RED)
│   │   └── state.py                # LangGraph pipeline state
│   │
│   ├── services/                   # Free AI & Processing Providers
│   │   ├── llm_factory.py          # Unified switch: Gemini Free API vs Local Ollama vs Groq
│   │   ├── extractor.py            # Document text/image parser & structured table extractor
│   │   ├── terminology.py          # Local LOINC matcher & range evaluator
│   │   └── verifier.py             # Rule-based & LLM fact-checking verifier
│   │
│   ├── graph/                      # Streamlined LangGraph Pipeline
│   │   ├── __init__.py
│   │   ├── state.py                # PipelineState TypedDict
│   │   ├── nodes.py                # 3 Core Agent nodes (extract, ground, reason_and_verify)
│   │   └── pipeline.py             # Compiled StateGraph with streaming support
│   │
│   ├── db/                         # Lightweight SQLite Layer
│   │   ├── __init__.py
│   │   ├── session.py              # SQLite connection (SQLAlchemy / aiosqlite)
│   │   └── models.py               # Reports, LabResults, Trends tables
│   │
│   └── api/                        # FastAPI REST & Streaming Layer
│       ├── __init__.py
│       ├── app.py                  # FastAPI app with CORS & error handlers
│       └── routes/
│           ├── reports.py          # Upload, analyze (with SSE streaming), get by ID
│           ├── trends.py           # Historical trends for biomarker tracking
│           └── health.py           # Health check & model status
│
└── tests/
    ├── test_extraction.py
    ├── test_terminology.py
    └── test_verifier.py
```

---

## 4. Free AI Model Strategy

This architecture is **100% free-tier and local-capable**:

| Provider | Model | Cost | When to Use | Setup |
|---|---|---|---|---|
| **Google Gemini API** *(Primary)* | `gemini-2.0-flash` / `gemini-1.5-flash` | **FREE** (Generous AI Studio quota: 15 RPM, 1M tokens/min) | Best overall balance of speed, multimodal vision, and medical reasoning | Get free key at [aistudio.google.com](https://aistudio.google.com) |
| **Local Ollama** *(Offline)* | `llama3.2:3b` / `qwen2.5:7b` / `phi-3` | **FREE** (Runs 100% offline on your CPU/GPU) | Complete privacy, no internet required, zero rate limits | Install Ollama & run `ollama run llama3.2` |
| **Groq API** *(Ultra-fast)* | `llama-3.3-70b-versatile` | **FREE** (Free daily quota) | Extremely fast token generation | Free API key from Groq Console |

The system uses `src/services/llm_factory.py` to seamlessly toggle between **Gemini**, **Ollama**, or **Groq** via a simple `.env` setting (`MODEL_PROVIDER=gemini` or `MODEL_PROVIDER=ollama`).

---

## 5. Streamlined 3-Step Agent Pipeline

Instead of 7 discrete micro-services with complex inter-agent RPCs, the pipeline runs 3 focused, deterministic-gated steps inside LangGraph:

```
[Uploaded Document: PDF / Image / Text]
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ 1. INGESTION & EXTRACTION NODE                         │
│ • Parses text/tables via pdfplumber or Gemini Vision   │
│ • Returns: [{test_name, value, unit, reference_range}] │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. TERMINOLOGY & FLAGGING NODE (Local & Deterministic) │
│ • Fuzzy-matches test names to `data/lab_ontology.json` │
│ • Assigns standardized LOINC & standard display names  │
│ • Evaluates value against biological reference ranges  │
│ • Computes Flag: [GREEN (Normal), AMBER, RED (Alert)]  │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. REASONING & VERIFIER NODE                           │
│ • Drafts plain-language patient summary                │
│ • Generates "Key Takeaways" & "Questions for Doctor"   │
│ • Verifier: Asserts no hallucinated numbers exist;     │
│   every claim is anchored in Step 1/Step 2 data        │
│ • Injects mandatory medical disclaimer                 │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
       [Complete Structured JSON Report + SSE Stream]
```

### 5.1 Local Medical Terminology (`data/lab_ontology.json`)
Contains pre-compiled reference data for standard lab panels:
- **Complete Blood Count (CBC)**: Hemoglobin, RBC, WBC, Platelets, MCV, Neutrophils, Lymphocytes...
- **Lipid Profile**: Total Cholesterol, HDL, LDL, Triglycerides, VLDL...
- **Metabolic / Kidney (KFT)**: Creatinine, Urea, BUN, eGFR, Uric Acid, Electrolytes...
- **Liver Function (LFT)**: SGOT/AST, SGPT/ALT, Bilirubin, Alkaline Phosphatase, Albumin...
- **Diabetes Panel**: Fasting Glucose, Postprandial Glucose, HbA1c...
- **Thyroid Panel**: T3, T4, TSH...
- **Vitamins**: Vitamin D, Vitamin B12, Iron, Ferritin...

Each entry contains standard units, normal min/max for male/female/general, LOINC code, and a layman explanation of what the test measures.

---

## 6. Core Data Models (Pydantic & SQLite)

### 6.1 `LabResult`
```python
class LabResult(BaseModel):
    test_name: str
    standard_name: Optional[str] = None
    loinc_code: Optional[str] = None
    observed_value: float | str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Literal["GREEN", "AMBER", "RED", "UNKNOWN"] = "UNKNOWN"
    explanation: Optional[str] = None  # Brief explanation of this specific marker
```

### 6.2 `ReportAnalysis`
```python
class ReportAnalysis(BaseModel):
    id: str
    report_title: str
    patient_info: Optional[Dict[str, Any]] = None
    results: List[LabResult]
    summary: str                     # Layman explanation (8th-grade reading level)
    key_findings: List[str]          # High-priority alerts or normal confirmations
    doctor_questions: List[str]      # 3-5 personalized questions for patient's doctor
    lifestyle_tips: List[str]        # General wellness/diet suggestions (non-prescriptive)
    confidence_score: float
    disclaimer: str
    created_at: datetime
```

---

## 7. API Endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/reports/analyze` | Accepts file upload or text. Runs pipeline and returns full JSON analysis. |
| `POST` | `/api/reports/analyze/stream` | Server-Sent Events (SSE) stream for live step-by-step progress & streaming summary. |
| `GET` | `/api/reports/history` | Returns list of previously analyzed reports stored in local SQLite. |
| `GET` | `/api/reports/{id}` | Returns a specific saved report. |
| `GET` | `/api/trends` | Returns time-series data for tracked biomarkers (e.g. HbA1c over past 6 months). |
| `GET` | `/api/health` | Returns health status and active AI model provider. |

---

## 8. How to Run Locally in 3 Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env (Use free Gemini key or local Ollama)
cp .env.example .env
# Set GEMINI_API_KEY=your_free_key (or MODEL_PROVIDER=ollama)

# 3. Start the server
python main.py
# Server runs on http://localhost:8000 (API docs at http://localhost:8000/docs)
```
