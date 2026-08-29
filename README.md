# Medical Report Analyzer

An intelligent, multi-agent AI framework that allows patients to upload lab and pathology reports and receive accurate, plain-language explanations with every claim traceable to source evidence and standard medical ontologies (LOINC, SNOMED CT).

## Features

- **Multi-Agent Architecture**: Specialized agents for ingestion, extraction, terminology grounding, clinical reasoning, and verification
- **Evidence-Grounded Explanations**: Every flagged value is traceable to source document spans and standard medical codes
- **Verification/Critic Agent**: Dedicated agent that checks generated explanations against coded evidence before release
- **Support for Multiple Formats**: PDF, images, and manually typed reports
- **Structured Extraction**: Test names, values, units, and reference ranges automatically extracted
- **Terminology Resolution**: Resolves findings to LOINC and SNOMED CT codes
- **Safety Gating**: Low-confidence claims are escalated for human review rather than presented to users
- **FHIR Compatible**: Generates FHIR Observation resources with proper coding

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **AI/LLM**: LangChain, LangGraph, Google Generative AI, Groq
- **Document Processing**: PDFPlumber, PyPDFium2, Pytesseract, Pillow
- **Database**: SQLAlchemy with SQLite
- **Report Generation**: ReportLab
- **Validation**: Pydantic

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- API keys for:
  - Google Generative AI (for Gemini API)
  - Groq (optional, for alternative LLM provider)
  - Anthropic (optional, for Claude models)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/medical-report-analyzer.git
cd medical-report-analyzer
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Server Configuration
DEBUG=true
HOST=localhost
PORT=8000

# Database
DATABASE_URL=sqlite:///./reports.db

# Security
JWT_SECRET=your_secret_key_here
```

Refer to `.env.example` for all available configuration options.

## Running the Application

### Start the Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

### Access the API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Run Tests

```bash
pytest
```

## API Endpoints

### Report Management

- **POST** `/api/v1/reports/upload` - Upload a report (PDF/image/typed)
- **POST** `/api/v1/reports/process` - Trigger the agent pipeline (SSE stream)
- **GET** `/api/v1/reports/{id}/status` - Get processing status
- **GET** `/api/v1/reports/{id}/result` - Get final analysis result
- **GET** `/api/v1/reports/{id}/audit` - Get agent decision audit trail
- **GET** `/api/v1/reports/trends` - Get trend data across multiple reports

### Review Queue

- **GET** `/api/v1/review/queue` - Get pending reviews for clinician
- **POST** `/api/v1/reports/{id}/review/resume` - Resume after human review

### Export

- **POST** `/api/v1/reports/{id}/export/pdf` - Generate downloadable PDF summary

### Health Check

- **GET** `/api/v1/health` - Server health status

## Project Structure

```
medical-report-analyzer/
├── src/
│   ├── api/                 # FastAPI application and routes
│   ├── db/                  # Database models and initialization
│   ├── graph/               # LangGraph agent orchestration
│   ├── schemas/             # Pydantic data models
│   ├── services/            # Business logic and utilities
│   └── static/              # Static assets
├── config/                  # Configuration settings
├── data/                    # Sample data and ontologies
│   ├── lab_ontology.json    # LOINC/SNOMED mappings
│   └── samples/             # Sample reports
├── .ai-reference/           # Project documentation
│   ├── PRD.md              # Product Requirements Document
│   ├── ARCHITECTURE.md     # System architecture
│   └── ARCHITECTURE_FRONTEND.md
├── .env.example            # Environment variables template
├── main.py                 # Application entrypoint
├── requirements.txt        # Python dependencies
├── pytest.ini              # Pytest configuration
└── README.md              # This file
```

## Architecture

The system uses a multi-agent orchestration pattern via LangGraph:

```
Upload → Ingestion Agent → Extraction Agent → Terminology Grounding Agent
                                                      ↓
                                          Clinical Reasoning Agent
                                                      ↓
                                         Verification/Critic Agent
                                                      ↓
                                    Escalation Queue (if needed)
                                                      ↓
                                            Explanation Agent
                                                      ↓
                                        Final Report & PDF
```

Each agent has:
- Specialized role and input/output types
- Confidence scoring
- Decision audit trail
- Integration with medical ontologies

For detailed architecture information, see [ARCHITECTURE.md](.ai-reference/ARCHITECTURE.md).

## Key Components

### 1. Ingestion Agent
Parses PDF/image reports and extracts structured text with layout awareness

### 2. Extraction Agent
Identifies and structures test results (name, value, unit, reference range)

### 3. Terminology Grounding Agent
Resolves findings to standard medical codes (LOINC, SNOMED CT)

### 4. Clinical Reasoning Agent
Generates plain-language explanations with clinical context

### 5. Verification/Critic Agent
**Core Research Contribution**: Validates explanations against coded evidence
- Detects unsupported claims
- Enforces escalation for low-confidence outputs
- Reduces hallucinations vs. single-pass LLM

### 6. Explanation Agent
Generates patient-facing summaries and exports

## Configuration

### Supported Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | false | Enable debug mode and hot reload |
| `HOST` | localhost | Server host address |
| `PORT` | 8000 | Server port |
| `DATABASE_URL` | sqlite:///./reports.db | Database connection URL |
| `JWT_SECRET` | - | Secret key for JWT authentication |
| `GEMINI_API_KEY` | - | Google Generative AI API key |
| `GROQ_API_KEY` | - | Groq API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |

## Development

### Install Dev Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
pytest -v
```

### Run Tests with Coverage

```bash
pytest --cov=src
```

## Documentation

- **Product Requirements**: [PRD.md](.ai-reference/PRD.md)
- **System Architecture**: [ARCHITECTURE.md](.ai-reference/ARCHITECTURE.md)
- **Frontend Architecture**: [ARCHITECTURE_FRONTEND.md](.ai-reference/ARCHITECTURE_FRONTEND.md)

## Data Models

### Report
```
- id: UUID
- user_id: UUID
- file_url: string
- file_type: PDF | IMAGE | TYPED
- status: UPLOADED | PROCESSING | COMPLETED | ESCALATED | FAILED
- created_at: datetime
- updated_at: datetime
```

### LabResult (Extracted)
```
- id: UUID
- report_id: UUID
- test_name: string
- observed_value: string
- unit: string (optional)
- reference_range: string (optional)
- flag: GREEN | AMBER | RED (optional)
- loinc_code: string (optional)
- snomed_code: string (optional)
- fhir_observation: FHIR Observation (JSON)
- confidence: float
- source_span: {start: int, end: int}
```

### AnalysisResult
```
- id: UUID
- report_id: UUID
- plain_language_summary: string
- flagged_findings: array of {test, flag, explanation}
- verification_status: APPROVED | REVISED | ESCALATED
- verification_confidence: float
- escalation_reason: string (optional)
- audit_trail: array of agent decisions
- created_at: datetime
```

## Safety & Privacy

⚠️ **Disclaimer**: This system is designed for educational and research purposes. It is NOT intended for clinical decision-making without medical professional review.

### Security Features

- TLS 1.2+ encryption in transit
- Field-level encryption for sensitive data
- Role-based access control
- Comprehensive audit logging
- Data retention policies with deletion on request

### Compliance

Designed with consideration for:
- HIPAA-style safeguards
- DPDP Act 2023 (India)
- ABDM/FHIR profile alignment

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Commit with clear messages: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature/your-feature`
6. Submit a pull request

## License

[Add your license here - MIT, Apache 2.0, etc.]

## Authors

**Swarup Futane** - Initial development

## Acknowledgments

- Medical ontologies: LOINC, SNOMED CT
- LLM providers: Google, Groq, Anthropic
- Document parsing: PDFPlumber, PyPDFium2, Pytesseract

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation in `.ai-reference/`
- Review API documentation at `/docs` when server is running

## Roadmap

### Current Release
- ✅ Multi-agent architecture with verification
- ✅ PDF/image report parsing
- ✅ Test extraction and terminology grounding
- ✅ Safety gating and escalation

### Planned Features
- ⏳ Frontend UI for patient portal
- ⏳ Trend analysis across multiple reports
- ⏳ Chest X-ray interpretation module
- ⏳ Multi-language support
- ⏳ B2B API with white-label options
- ⏳ Kubernetes deployment templates

---

**Status**: Under Active Development  
**Last Updated**: August 2026
