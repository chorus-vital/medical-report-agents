"""
Document text / image parser & structured table extractor.

Extraction flow
================
  PDF   → pdfplumber (text + tables) → LLM structured extraction
          ↳ scanned / image-only?    → page images → Gemini Vision
  Image → Gemini Vision multimodal   → structured JSON
  Text  → Direct LLM structured extraction
          ↳ LLM fails?              → regex fallback

Every path converges on a common output:
  (extracted_items: list[dict], patient_info: dict | None)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docx
import pdfplumber
from PIL import Image

from config.settings import settings
from src.services.llm_factory import get_chat_model, get_vision_model

logger = logging.getLogger(__name__)


# ─────────────────────────── Prompt Templates ───────────────────────────────

EXTRACTION_PROMPT = """\
You are a precise medical lab report parser.  Your ONLY job is to extract
ALL lab test results and patient information from the text below.

─── OUTPUT FORMAT (strict JSON, nothing else) ───
{{
  "patient_info": {{
    "name": "<string or null>",
    "age": "<string or null>",
    "sex": "<Male | Female | null>",
    "date": "<report date string or null>",
    "lab_name": "<laboratory name or null>",
    "referring_doctor": "<doctor name or null>"
  }},
  "lab_results": [
    {{
      "test_name": "<name exactly as printed>",
      "observed_value": "<result value exactly as printed>",
      "unit": "<unit or null>",
      "reference_range": "<normal range as printed or null>"
    }}
  ]
}}

─── RULES ───
• Extract EVERY test — even normal-looking ones.
• If a panel has sub-tests (e.g. CBC → Hemoglobin, WBC, …), list each
  sub-test as its own object.
• Keep observed_value verbatim (e.g. "14.5", "Positive", "<0.5").
• Do NOT invent or guess any value.
• Return ONLY the JSON object — no markdown fences, no commentary.

─── LAB REPORT TEXT ───
{text}
"""

VISION_PROMPT = """\
You are a precise medical lab report parser.  Look at this lab report image
and extract ALL information.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "patient_info": {{
    "name": "<string or null>",
    "age": "<string or null>",
    "sex": "<Male | Female | null>",
    "date": "<report date or null>",
    "lab_name": "<laboratory name or null>"
  }},
  "lab_results": [
    {{
      "test_name": "<test name>",
      "observed_value": "<result value>",
      "unit": "<unit or null>",
      "reference_range": "<reference range or null>"
    }}
  ]
}}

Rules: extract ALL tests, keep values exactly as printed, do NOT invent values.\
"""


# ──────────────────────────── Public API ────────────────────────────────────

async def extract_from_file(
    file_path: str,
    file_type: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Main entry-point — extract lab results + patient info from a file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the uploaded file.
    file_type : str
        ``"pdf"`` | ``"image"`` | ``"text"``

    Returns
    -------
    (extracted_items, patient_info)
        *extracted_items* is a list of dicts, each with keys
        ``test_name``, ``observed_value``, ``unit``, ``reference_range``.
        *patient_info* may be ``None`` if nothing was found.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_type == "pdf":
        return await _extract_from_pdf(path)
    if file_type == "image":
        return await _extract_from_image(path)
    if file_type == "docx":
        return await _extract_from_docx(path)
    if file_type == "text":
        return await _extract_from_text(path.read_text(encoding="utf-8"))

    raise ValueError(
        f"Unsupported file_type: '{file_type}'. Use 'pdf', 'image', 'docx', or 'text'."
    )


def detect_file_type(filename: str) -> str:
    """Infer ``file_type`` from a filename extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        return "image"
    if ext in {".docx", ".doc"}:
        return "docx"
    if ext in {".txt", ".csv", ".text"}:
        return "text"
    raise ValueError(f"Unsupported file extension: '{ext}'")


# ───────────────────────── DOCX Extraction ──────────────────────────────────

async def _extract_from_docx(
    path: Path,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extract lab data from a Word (.docx) file by reading paragraphs + tables."""
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        logger.error("Failed to open .docx file: %s", exc)
        return [], None

    # ── extract paragraphs ──
    paragraphs_text = "\n".join(
        p.text for p in document.paragraphs if p.text.strip()
    )

    # ── extract tables ──
    tables_text = ""
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            tables_text += " | ".join(cells) + "\n"

    combined = paragraphs_text.strip()
    if tables_text.strip():
        combined += "\n\n--- EXTRACTED TABLES ---\n" + tables_text.strip()

    if len(combined.strip()) > 50:
        logger.info(
            "DOCX text extraction OK (%d para-chars, %d table-chars)",
            len(paragraphs_text),
            len(tables_text),
        )
        return await _extract_from_text(combined)

    logger.warning("DOCX appears empty or too short (%d chars)", len(combined))
    return [], None


# ───────────────────────── PDF Extraction ───────────────────────────────────

async def _extract_from_pdf(
    path: Path,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse a PDF with pdfplumber; fall back to Gemini Vision for scans."""
    raw_text = ""
    tables_text = ""

    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                # --- plain text ---
                page_text = page.extract_text() or ""
                raw_text += page_text + "\n"

                # --- tables (often more reliable for structured lab data) ---
                for table in page.extract_tables():
                    for row in table:
                        if row:
                            cells = [
                                str(c).strip() if c else "" for c in row
                            ]
                            tables_text += " | ".join(cells) + "\n"
    except Exception as exc:
        logger.warning("pdfplumber failed: %s", exc)

    combined = raw_text.strip()
    if tables_text.strip():
        combined += "\n\n--- EXTRACTED TABLES ---\n" + tables_text.strip()

    # If we got meaningful text → parse with LLM
    if len(combined.strip()) > 50:
        logger.info(
            "PDF text extraction OK (%d chars, %d table-chars)",
            len(raw_text),
            len(tables_text),
        )
        return await _extract_from_text(combined)

    # Scanned / image-only PDF → Gemini Vision fallback
    logger.info("PDF appears scanned — falling back to Gemini Vision")
    return await _extract_pdf_via_vision(path)


async def _extract_pdf_via_vision(
    path: Path,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Render each PDF page as an image and pass to Gemini Vision."""
    all_items: List[Dict[str, Any]] = []
    patient_info: Optional[Dict[str, Any]] = None

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            logger.info("Vision-extracting page %d …", i + 1)
            img = page.to_image(resolution=200).original
            items, info = await _extract_from_pil_image(img)
            all_items.extend(items)
            if info and patient_info is None:
                patient_info = info

    return all_items, patient_info


# ──────────────────────── Image Extraction ──────────────────────────────────

async def _extract_from_image(
    path: Path,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extract lab data from an image file via Gemini Vision."""
    img = Image.open(str(path))
    # Convert to RGB if needed (handles RGBA PNGs, palette images, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return await _extract_from_pil_image(img)


async def _extract_from_pil_image(
    img: Image.Image,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Send a PIL image to Gemini Vision and parse the JSON response."""
    try:
        from google.genai import types as genai_types

        client = get_vision_model()  # returns google.genai.Client
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[VISION_PROMPT, img],
            config=genai_types.GenerateContentConfig(temperature=0.1),
        )
        return _parse_llm_json(response.text)
    except Exception as exc:
        logger.error("Gemini Vision extraction failed: %s", exc)
        return [], None


# ────────────────────── Text → LLM Extraction ──────────────────────────────

async def _extract_from_text(
    raw_text: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run the structured-extraction prompt through the configured LLM."""
    # Truncate very long documents to stay within free-tier token limits
    MAX_CHARS = 15_000
    if len(raw_text) > MAX_CHARS:
        raw_text = raw_text[:MAX_CHARS] + "\n… [truncated]"

    try:
        llm = get_chat_model()
        prompt = EXTRACTION_PROMPT.format(text=raw_text)
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # Gemini 3.6+ may return content as a list of parts
        if isinstance(content, list):
            text = "".join(
                part if isinstance(part, str) else part.get("text", str(part))
                for part in content
            )
        else:
            text = str(content)
        return _parse_llm_json(text)
    except Exception as exc:
        logger.error("LLM text extraction failed: %s — trying regex fallback", exc)
        return _regex_fallback(raw_text), _regex_patient_info(raw_text)


# ───────────────────── JSON Response Parser ─────────────────────────────────

def _parse_llm_json(
    response_text: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Robustly parse the JSON that the LLM returns.

    Handles markdown fences, stray text around the JSON object, etc.
    """
    cleaned = response_text.strip()
    # Strip ```json … ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    data: dict = {}
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to isolate the outermost { … }
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Could not parse JSON from LLM response")
                return [], None
        else:
            logger.error("No JSON object found in LLM response")
            return [], None

    lab_results = data.get("lab_results", [])
    patient_info = data.get("patient_info")

    # ── clean each lab item ──
    clean_items: List[Dict[str, Any]] = []
    for item in lab_results:
        if not isinstance(item, dict) or not item.get("test_name"):
            continue
        clean_items.append(
            {
                "test_name": str(item["test_name"]).strip(),
                "observed_value": str(item.get("observed_value", "")).strip(),
                "unit": _nullify(item.get("unit")),
                "reference_range": _nullify(item.get("reference_range")),
            }
        )

    # ── clean patient_info ──
    if isinstance(patient_info, dict):
        patient_info = {
            k: _nullify(v) for k, v in patient_info.items()
        }
    else:
        patient_info = None

    logger.info("Parsed %d lab items from LLM response", len(clean_items))
    return clean_items, patient_info


# ────────────────────── Regex Fallback Extractor ────────────────────────────

_LAB_LINE = re.compile(
    r"^"
    r"(?P<name>.{3,45}?)"                          # test name
    r"\s{2,}"                                       # wide whitespace separator
    r"(?P<value>[\d.,]+|[A-Za-z]+(?:\s[A-Za-z]+)?)" # numeric or text value
    r"\s+"
    r"(?P<unit>[a-zA-Z/%µ·^0-9]+(?:/[a-zA-Z]+)?)"  # unit
    r"\s+"
    r"(?P<range>"
    r"[\d.,]+\s*[-–—]\s*[\d.,]+"                    # range: 12.0 - 16.0
    r"|[<>≤≥]\s*[\d.,]+"                            # inequality: < 200, > 90
    r")"
    r"\s*$",
    re.MULTILINE,
)


def _regex_fallback(raw_text: str) -> List[Dict[str, Any]]:
    """
    Best-effort regex extractor for tabular lab-report formats::

        Hemoglobin          14.5       g/dL       12.0 - 16.0
    """
    results: List[Dict[str, Any]] = []
    for m in _LAB_LINE.finditer(raw_text):
        results.append(
            {
                "test_name": m.group("name").strip().rstrip(":"),
                "observed_value": m.group("value").strip(),
                "unit": m.group("unit").strip(),
                "reference_range": m.group("range").strip(),
            }
        )
    logger.info("Regex fallback extracted %d items", len(results))
    return results


_PAT_NAME = re.compile(
    r"(?:patient\s*name|name)\s*[:\-]\s*(.+?)(?:\n|$)", re.I
)
_PAT_AGE = re.compile(
    r"(?:age)\s*[:\-]\s*(\d{1,3})\s*(?:years?|yrs?)?", re.I
)
_PAT_SEX = re.compile(
    r"(?:sex|gender)\s*[:\-]\s*(male|female|m|f)\b", re.I
)


def _regex_patient_info(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract patient demographics with simple regex patterns."""
    info: Dict[str, Any] = {}

    m = _PAT_NAME.search(raw_text)
    if m:
        info["name"] = m.group(1).strip()

    m = _PAT_AGE.search(raw_text)
    if m:
        info["age"] = m.group(1).strip()

    m = _PAT_SEX.search(raw_text)
    if m:
        raw = m.group(1).strip().upper()
        info["sex"] = "Male" if raw in ("M", "MALE") else "Female"

    return info or None


# ──────────────────────────── Helpers ───────────────────────────────────────

def _nullify(value: Any) -> Optional[str]:
    """Convert empty / sentinel strings to ``None``."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ("", "null", "none", "n/a", "-", "--"):
        return None
    return s
