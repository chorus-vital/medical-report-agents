# Product Requirements Document (PRD)
## Agentic AI Framework for Intelligent Medical Report Analysis

**Version:** 1.0  
**Date:** August 2026  
**Domain:** Health Informatics / Clinical NLP  
**Deliverables:** Working System (2 repositories) + Peer-Reviewed Research Paper

---

## 1. Executive Summary

This project builds a **multi-agent AI framework** that allows patients to upload lab/pathology reports and receive accurate, plain-language, evidence-grounded explanations — with every claim traceable to source evidence and standard medical ontologies (LOINC, SNOMED CT).

Unlike a linear OCR → NLP → summary pipeline, this system uses **specialized, tool-using agents** coordinated by a state-machine orchestrator, with a dedicated **verification/critic agent** that can reject and re-route low-confidence output before it reaches the patient.

### One-Sentence Contribution Statement

> *"We present an agentic, terminology-grounded framework for consumer lab-report interpretation in which a dedicated verifier agent checks generated explanations against coded clinical evidence before release, and we measure how much this reduces unsupported claims versus a single-pass LLM pipeline on a corpus of real-world Indian diagnostic reports."*

---

## 2. Problem Statement

- Patients receive lab reports with complex medical terminology they cannot interpret.
- Single-pass LLM summarization is risky — hallucinated clinical claims can cause panic or false reassurance.
- Existing agentic medical systems target EHR question-answering or radiology, not **consumer-uploaded lab/pathology PDFs**.
- Indian diagnostic reports (SRL, Thyrocare, Dr Lal PathLabs) are heterogeneous, mixing English with regional-language patient sections.

---

## 3. System Repositories

The system is split into **two repositories**:

| Repository | Purpose | Tech Stack |
|---|---|---|
| **`medical-report-agents`** (this repo) | Multi-agent backend: orchestration, all AI agents, FastAPI, database, FHIR services | Python, LangGraph, FastAPI, PostgreSQL, Anthropic API |
| **`medical-report-frontend`** (future) | Patient-facing web UI: upload, report view, trend charts, reviewer queue, PDF export | React, TypeScript, Vite, TailwindCSS |

---

## 4. Novelty & Research Gaps Addressed

| Gap | Description |
|---|---|
| **Gap 1: Consumer lab reports, not EHRs** | Patient-facing, self-uploaded lab/pathology PDFs — messy scans, mixed formats, multiple labs, no structured source system |
| **Gap 2: Terminology-grounded explanation** | Dedicated Terminology Grounding Agent resolves every finding to LOINC/SNOMED CT codes before reasoning — creating auditable, FHIR-compatible records |
| **Gap 3: Explicit verifier/critic agent** | Verification Agent cross-checks reasoning against coded evidence; forces escalation for ungrounded claims (headline ablation metric) |
| **Gap 4: Indian-market grounding** | Evaluation on Indian report formats; alignment with ABDM/FHIR profiles |

---

## 5. Personas

| ID | Persona | Need |
|---|---|---|
| P1 | **Self-tracking patient** | Understand blood work without waiting for a follow-up appointment |
| P2 | **Caregiver** | Manage elderly parent's reports across multiple labs and visits |
| P3 | **Time-pressed clinician** | Fast, coded summary before a consult — not a substitute for judgment |
| P4 | **Diagnostic lab / insurer (B2B)** | API for automated pre-screening or claim triage |

---

## 6. Goals

- Let a patient upload a lab/pathology report and get an accurate, plain-language, evidence-grounded explanation within seconds.
- Every flagged value must be traceable to a source span in the document and a standard terminology code.
- Never let an unverified or low-confidence claim reach the user without a "consult your doctor" escalation.

---

## 7. Non-Goals (Out of Scope for MVP)

- Diagnosis, prescription, or treatment recommendations of any kind.
- Full radiology reporting (beyond the optional bounded X-ray stretch module).
- EHR integration with live hospital systems (design for FHIR compatibility, don't build the integration).
- Training custom imaging models from scratch.

---

## 8. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Accept PDF, image, and manually typed report input | **MVP** |
| FR-2 | Extract test name, value, unit, reference range per record | **MVP** |
| FR-3 | Resolve each test to LOINC code and each abnormal finding to SNOMED CT | **MVP** |
| FR-4 | Green/amber/red flagging against age/sex-adjusted reference ranges | **MVP** |
| FR-5 | Plain-language explanation with no unsupported claims (verifier-gated) | **MVP** |
| FR-6 | Trend view across multiple reports over time | **MVP** |
| FR-7 | Downloadable PDF summary for a doctor visit | **MVP** |
| FR-8 | Escalation banner + reviewer queue for low-confidence or critical-value cases | **MVP** |
| FR-9 | Chest X-ray findings module | **Stretch** |
| FR-10 | B2B API with white-label output schema | **Stretch** |

---

## 9. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Accuracy** | ≥95% extraction F1 on held-out labeled report set; verifier catches ≥90% of injected unsupported claims |
| **Latency** | End-to-end under 20s for a single-page report (excluding cold model load) |
| **Privacy** | Encrypted at rest and in transit; no report content used for third-party model training; deletable on request |
| **Auditability** | Every agent decision logged with input/output and confidence score, replayable end-to-end |
| **Safety** | Persistent "not a medical diagnosis" disclaimer; hard block on prescriptive language in reasoning agent's system prompt |
| **Accessibility** | Plain-language output at ~8th-grade reading level; multi-language support (English + one Indian regional language) as stretch |

---

## 10. Agent Definitions

### 10.1 Agent Pipeline Overview

```
Patient Upload --> [1. Ingestion Agent] --> [2. Extraction Agent] --> [3. Terminology Grounding Agent]
                                                                              |
              [7. Explanation Agent] <-- [6. Verification/Critic Agent] <-- [4. Clinical Reasoning Agent]
                      |                          | (re-route/escalate)
               Final Report/PDF          [Human-in-the-Loop Queue]
```

### 10.2 Agent Specifications

| # | Agent | Input | Output (typed) | Primary Tool/Model |
|---|---|---|---|---|
| 1 | **Ingestion / Vision Agent** | Raw PDF, image, scan | Layout-aware Markdown + bounding boxes | VLM document parser (Docling / Gemini) |
| 2 | **Extraction Agent** | Parsed document | Structured `{test, value, unit, ref_range}` records | Clinical NER transformer (PubMedBERT/BioClinicalBERT) |
| 3 | **Terminology Grounding Agent** | Extracted records | FHIR `Observation` with LOINC/SNOMED codes | Entity-resolution via FHIR terminology service |
| 4 | **Clinical Reasoning Agent** | Coded observations + history | Flag (green/amber/red) + plain-language draft | Claude Sonnet/Opus via Anthropic API + RAG |
| 5 | **Imaging Agent** *(stretch)* | Chest X-ray image | Findings text + region grounding | Open chest-X-ray VLM (frozen, wrapped) |
| 6 | **Verification / Critic Agent** | Draft explanation + coded evidence | Approve / revise / escalate decision | LLM-as-judge + rule-based checks |
| 7 | **Explanation Agent** | Verified content | Patient-facing summary, PDF, trend chart | Templated generation, no free re-reasoning |

### 10.3 Orchestrator

- **Type:** LangGraph state-machine graph (not a single prompt)
- **Responsibilities:** Route documents through agents, enforce confidence gating, manage re-route loops from verifier, trigger human-in-the-loop escalation
- **Key behavior:** Verifier can reject output and re-route back to reasoning agent (max 2 retries before forced escalation)

---

## 11. API Endpoints (Agents Repository)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/reports/upload` | Upload a report (PDF/image/typed) |
| `POST` | `/api/v1/reports/process` | Trigger the full agent pipeline (returns SSE stream) |
| `GET` | `/api/v1/reports/{id}/status` | Get processing status |
| `GET` | `/api/v1/reports/{id}/result` | Get final analysis result |
| `GET` | `/api/v1/reports/{id}/audit` | Get full agent decision audit trail |
| `GET` | `/api/v1/reports/trends` | Get trend data across multiple reports for a user |
| `POST` | `/api/v1/reports/{id}/review/resume` | Resume after human-in-the-loop review |
| `GET` | `/api/v1/review/queue` | Get pending reviews for clinician queue |
| `POST` | `/api/v1/reports/{id}/export/pdf` | Generate downloadable PDF summary |
| `GET` | `/api/v1/health` | Health check |

---

## 12. Data Models (Core)

### 12.1 Report
```
Report {
  id: UUID
  user_id: UUID
  file_url: str
  file_type: enum(PDF, IMAGE, TYPED)
  status: enum(UPLOADED, PROCESSING, COMPLETED, ESCALATED, FAILED)
  raw_text: str (parsed)
  created_at: datetime
  updated_at: datetime
}
```

### 12.2 LabResult (Extracted)
```
LabResult {
  id: UUID
  report_id: UUID
  test_name: str
  observed_value: str
  unit: str | null
  reference_range: str | null
  flag: enum(GREEN, AMBER, RED) | null
  loinc_code: str | null
  snomed_code: str | null
  fhir_observation: JSONB   // Full FHIR Observation resource
  confidence: float
  source_span: {start: int, end: int}
}
```

### 12.3 AnalysisResult
```
AnalysisResult {
  id: UUID
  report_id: UUID
  plain_language_summary: str
  flagged_findings: []{test, flag, explanation}
  verification_status: enum(APPROVED, REVISED, ESCALATED)
  verification_confidence: float
  escalation_reason: str | null
  audit_trail: JSONB[]     // Array of {agent, input_hash, output, confidence, latency_ms}
  created_at: datetime
}
```

---

## 13. Success Metrics

| Metric | Description | Target |
|---|---|---|
| **Extraction F1** | F1 on test/value/unit/range extraction vs. manually annotated ground truth | >= 0.95 |
| **Grounding Precision** | % of LOINC/SNOMED mappings judged correct by clinician reviewer | >= 0.90 |
| **Hallucination Rate** | Unsupported claims per 100 generated explanations, with vs. without verifier | >= 50% reduction |
| **Escalation Precision/Recall** | Does the system escalate the right cases, and only those | Precision >= 0.85, Recall >= 0.90 |
| **Time-to-Understanding** | User study — self-reported comprehension before/after using the tool | Significant improvement |

---

## 14. Security & Compliance

| Area | Requirement |
|---|---|
| **Encryption** | TLS 1.2+ in transit; AES-256 at rest; field-level encryption for PII |
| **Auth** | JWT-based (Firebase Auth or self-hosted); role-based access for reviewer queue |
| **India (DPDP Act 2023)** | Consent and data-minimization; ABDM/FHIR profile alignment |
| **HIPAA-style safeguards** | Design benchmark even without formal certification |
| **Data retention** | Configurable retention/deletion policy; deletable on request |
| **IRB/Ethics** | Required if real patient reports used for evaluation |

---

## 15. Deployment

| Environment | Purpose | Notes |
|---|---|---|
| **Dev** | Local + shared dev | Synthetic/de-identified sample reports only |
| **Staging** | End-to-end demo, user study, paper evaluation | Mirrors prod config; evaluation metrics captured here |
| **Prod** *(optional, post-capstone)* | Public or pilot deployment | Only if compliance review complete |

### Infrastructure
- Containerized services (Docker) per agent/service
- Docker Compose for capstone scale; Kubernetes for B2B stretch
- Encrypted object storage for uploaded documents
- Structured logging per agent call (input hash, output, confidence, latency)

---

## 16. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| LLM hallucinated clinical claim reaches user | **High** | Verifier agent + hard escalation rule + disclaimer layer + human review queue |
| Poor OCR on low-quality scans skews downstream | **Medium** | Confidence score from parsing agent gates pipeline; low-confidence -> manual review |
| Terminology mis-mapping (wrong LOINC/SNOMED) | **Medium** | Confidence threshold on resolver; below-threshold flagged, not silently accepted |
| Scope creep from imaging module | **Low** | Keep imaging as wrapped external model behind tool interface; no from-scratch training |

---

## 17. Timeline (6 Phases, 16-20 weeks)

| Phase | Weeks | Focus |
|---|---|---|
| **Phase 1** | 1-2 | Scoping, literature grounding, architecture finalization |
| **Phase 2** | 3-6 | Core pipeline (Agents 1-4): Ingestion, Extraction, Grounding, Reasoning |
| **Phase 3** | 7-9 | Verifier agent & safety loop (core research contribution) |
| **Phase 4** | 10-12 | Frontend, trend view, PDF export |
| **Phase 5** | 13-15 | Evaluation, stretch module, user study |
| **Phase 6** | 16-18 | Paper writing & submission |
