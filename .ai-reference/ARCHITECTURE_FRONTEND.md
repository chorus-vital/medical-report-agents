# Simplified Architecture Document — Frontend Web App
## `medical-report-frontend` (Localhost & Free-Tier Edition)

**Version:** 2.0 (Simplified for Localhost & Fast Development)  
**Date:** August 2026  
**Stack:** React 19 / 18 · TypeScript · Vite · TailwindCSS · Lucide Icons · Recharts

---

## 1. Design Philosophy

The frontend is designed as a **clean, intuitive, single-page application (SPA)** that runs locally in seconds (`npm run dev`). It focuses on **consumer clarity** — transforming opaque laboratory numbers into color-coded visual cards, normal range meters, plain-language summaries, and actionable questions for doctor appointments.

| Feature | Original Over-Engineered Architecture | Simplified Localhost Architecture |
|---|---|---|
| **Authentication** | Multi-role JWT + Firebase Auth + RBAC Guards | **Local-first / Direct Access** (No login wall for local testing; optional profile switcher) |
| **State Management** | Complex TanStack Query + multiple Zustand slices | **Lightweight React Query / Custom Hooks** for fetching and caching report data |
| **Real-time Pipeline** | Complex multi-event SSE protocol | **Streamlined SSE Hook (`useReportStream`)** for real-time progress & token streaming |
| **PDF Generation** | Complex server-side WeasyPrint with signed URLs | **Browser-Native Print CSS (`window.print()`) & Client-side PDF export** |
| **Model Configuration** | Hardcoded backend-only environment config | **In-App Model & Provider Switcher** (Gemini Free Key / Local Ollama / Groq) |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              REACT APPLICATION (http://localhost:5173)          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    MAIN VIEWS / TABS                     │  │
│  │  [1. Upload & Analyze]   [2. Report Dashboard]           │  │
│  │  [3. Biomarker Trends]   [4. History & Archive]          │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │                   UI COMPONENT LAYER                     │  │
│  │  • UploadDropzone (PDF/Image/Text/Sample Reports)        │  │
│  │  • LivePipelineStepper (Extract -> Ground -> Verify)     │  │
│  │  • LabResultsTable (Color Badges + Range Visualizer)     │  │
│  │  • LaymanSummaryCard ("What This Means for You")         │  │
│  │  • DoctorQuestionsList ("Questions to Ask Your Doctor")  │  │
│  │  • TrendLineChart (Recharts Historical Graph)            │  │
│  │  • ClinicalInspectorDrawer (LOINC codes & Audit Trail)   │  │
│  │  • MedicalDisclaimerBanner (Safety & DPDP notice)        │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼───────────────────────────────┐  │
│  │                   HOOKS & API CLIENT                     │  │
│  │  • useReportAnalysis (Upload & Trigger LangGraph stream) │  │
│  │  • useBiomarkerTrends (Historical trend chart data)      │  │
│  │  • useSettings (Gemini API key & Ollama URL settings)    │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                 FastAPI Backend (http://localhost:8000)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Folder Structure

```
medical-report-frontend/
├── .env.example                    # VITE_API_BASE_URL=http://localhost:8000
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── README.md
│
├── public/
│   └── favicon.ico
│
└── src/
    ├── main.tsx                    # React entry point
    ├── App.tsx                     # Main layout & navigation tabs
    │
    ├── api/                        # Backend API communication
    │   ├── client.ts               # Axios / Fetch base client
    │   └── reports.ts              # uploadReport, getReport, getTrends, streamAnalysis
    │
    ├── hooks/                      # Custom React Hooks
    │   ├── useReportStream.ts      # SSE connection for live step progress & streaming text
    │   ├── useTrends.ts            # Fetch historical biomarker trends
    │   └── useSettings.ts          # LocalStorage settings (API keys, active provider)
    │
    ├── components/
    │   ├── layout/
    │   │   ├── Header.tsx          # App title, provider indicator, settings modal trigger
    │   │   └── Disclaimer.tsx      # Persistent "Not a medical diagnosis" disclaimer
    │   │
    │   ├── upload/
    │   │   ├── UploadDropzone.tsx  # Drag & drop PDF/image file upload
    │   │   ├── TextInputModal.tsx  # Paste raw lab text directly
    │   │   └── SampleReports.tsx   # 1-click test reports (CBC, Lipid, Thyroid)
    │   │
    │   ├── processing/
    │   │   └── PipelineStepper.tsx # Visual progress: Ingestion -> Grounding -> Verification
    │   │
    │   ├── report/
    │   │   ├── ReportSummary.tsx   # Plain-language explanation & key findings card
    │   │   ├── LabResultCard.tsx   # Individual test card with range meter
    │   │   ├── LabResultTable.tsx  # Full table with Green/Amber/Red flag badges
    │   │   ├── RangeVisualizer.tsx # Visual horizontal bar showing where value sits in range
    │   │   ├── DoctorQuestions.tsx # Actionable list of questions for doctor consult
    │   │   └── ClinicalDrawer.tsx  # Collapsible audit inspector (LOINC codes & raw text)
    │   │
    │   ├── trends/
    │   │   └── TrendChart.tsx      # Recharts line chart (e.g. Glucose, HbA1c over time)
    │   │
    │   ├── settings/
    │   │   └── SettingsModal.tsx   # Toggle Gemini Free API / Local Ollama / Groq
    │   │
    │   └── common/
    │       ├── Button.tsx
    │       ├── Badge.tsx
    │       └── LoadingSpinner.tsx
    │
    ├── types/                      # TypeScript definitions
    │   ├── report.ts               # LabResult, ReportAnalysis, FlagType
    │   └── sse.ts                 # SSE Stream event payloads
    │
    └── utils/
        ├── formatters.ts           # Format dates, numbers, units
        └── rangeCalculator.ts      # Calculate percentage position on normal range bar
```

---

## 4. Key UI Components & Interactions

### 4.1 Quick Upload & Sample Loader
- Drag & drop PDF, JPG, PNG, or paste text directly.
- **"Try a Sample Report" buttons**: Includes 3 pre-loaded samples (e.g. *Complete Blood Count with Anemia*, *Lipid Panel with Elevated LDL*, *Thyroid Profile*) so developers/evaluators can test with 1 click without finding a PDF.

### 4.2 Real-time Pipeline Stepper
While the LangGraph backend processes the report, the UI displays real-time progress indicators:
1. `Ingestion & OCR`: Extracting text and tabular rows.
2. `Terminology Grounding`: Matching to standard LOINC dictionary and reference ranges.
3. `Clinical Reasoning & Verification`: Generating summary and fact-checking all numbers.
4. `Done`: Instant smooth transition to the interactive dashboard.

### 4.3 Interactive Lab Result Card & Range Visualizer
Instead of static numbers, each biomarker displays a visual horizontal range meter:

```
[ LOW ] ─────── [ NORMAL RANGE: 13.0 - 17.0 g/dL ] ─────── [ HIGH ]
                  ▲ (Your Value: 11.2 - Amber)
```

- **Green**: Value is well within normal biological limits.
- **Amber**: Value is slightly borderline (elevated or low); warrants lifestyle monitoring or discussion with physician.
- **Red**: Significantly abnormal; highlighted at the top of the summary for priority doctor consultation.

### 4.4 Plain-Language Explanation & Doctor Questions
- **"What this means for you"**: Generated at an 8th-grade reading level, explaining complex medical jargon in clear terms.
- **"Questions to ask your doctor"**: 3-5 specific questions tailored to any abnormal values (e.g. *"Should we check my serum ferritin given my low hemoglobin?"*).

### 4.5 Historical Biomarker Trends
- Uses **Recharts** to plot time-series lines for repeat tests (e.g., Fasting Glucose, HbA1c, Total Cholesterol, Creatinine).
- Displays historical points with colored dots matching the flag status on each date.

### 4.6 1-Click Doctor Visit PDF Export / Print
- Features a clean print stylesheet (`@media print`) that formats the report cleanly on standard A4 / Letter pages.
- Allows users to click "Print / Save PDF" to take a clean physical or digital copy to their physician appointment.

---

## 5. Technology Stack Summary

| Layer | Choice | Why |
|---|---|---|
| **Framework** | React 19 / 18 | Component ecosystem, fast rendering, widespread familiarity |
| **Build Tool** | Vite | Lightning-fast local startup (<300ms) & hot module reloading |
| **Styling** | TailwindCSS | Rapid styling, responsive utility classes |
| **Icons** | Lucide React | Clean, modern medical & UI icon set |
| **Charts** | Recharts | Declarative, smooth React charting for biomarker trends |
| **HTTP & Streaming** | Native `fetch` + `ReadableStream` | Zero external dependency needed for SSE streaming |
| **PDF Generation** | Native CSS `@media print` | Zero bloat, perfectly crisp vector printouts directly from browser |

---

## 6. How to Run Locally in 2 Steps

```bash
# 1. Install dependencies
npm install

# 2. Start development server
npm run dev
# App runs on http://localhost:5173 (auto-proxies API requests to http://localhost:8000)
```
