"""
Robust PDF text extraction.

Why this module exists
======================
Many real-world lab reports (Orange Health, Thyrocare, Metropolis, …) embed
subsetted fonts with broken/absent ``ToUnicode`` maps.  ``pdfplumber`` decodes
those glyphs as ``\\x00``, which destroys exactly the characters that matter
most for parsing::

    pdfplumber : 'Red Blood Cells \\x00RBC\\x00 Count 5.47 4.5 \\x00 5.5'
    pypdfium2  : 'Red Blood Cells (RBC) Count\\n5.47 4.5 - 5.5'

The range separator ``-`` and the parentheses around the abbreviation are gone
in the first case, so both the LLM and the regex fallback lose information.

Strategy
--------
Extract with **both** engines, score each result, and return the better one.
``pypdfium2`` (PDFium — the same engine Chrome uses) usually wins on these
files; ``pdfplumber`` is kept because it recovers ruled tables that PDFium
flattens.  Table text from pdfplumber is appended as a supplementary block.

A small repair pass fixes residual mojibake in units (``mm\\ufffd`` → ``mm³``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# pdfminer is extremely chatty about broken font descriptors on these files;
# the warnings are harmless and drown out real log output.
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)


# ────────────────────────── Mojibake repair ─────────────────────────────────

# Unit glyphs that survive extraction as U+FFFD / \x00. Ordered longest-first so
# the compound patterns win before the single-character ones.
_UNIT_REPAIRS: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"10[�\x00]\s*/\s*[�\x00]L"), "10³/µL"),
    (re.compile(r"10[�\x00]\s*/\s*[�\x00]l"), "10³/µl"),
    (re.compile(r"\bmm[�\x00]"), "mm³"),
    (re.compile(r"\bcm[�\x00]"), "cm³"),
    (re.compile(r"\bdl[�\x00]", re.I), "dL"),
    (re.compile(r"[�\x00]L\b"), "µL"),
    (re.compile(r"[�\x00]mol"), "µmol"),
    (re.compile(r"[�\x00]g\b"), "µg"),
    (re.compile(r"[�\x00]IU"), "µIU"),
    (re.compile(r"10[�\x00]"), "10³"),
]

_MOJIBAKE_CHARS = "�\x00"


def repair_mojibake(text: str) -> str:
    """Fix unit superscripts / micro-signs that lost their Unicode mapping."""
    for pattern, replacement in _UNIT_REPAIRS:
        text = pattern.sub(replacement, text)
    # Anything still unmapped becomes a space rather than a control character,
    # so downstream regexes and the LLM see a clean word boundary.
    return re.sub(f"[{_MOJIBAKE_CHARS}]", " ", text)


def _mojibake_ratio(text: str) -> float:
    """Fraction of characters that failed to decode (0.0 = clean)."""
    if not text:
        return 1.0
    return sum(text.count(c) for c in _MOJIBAKE_CHARS) / len(text)


# ─────────────────────────────── Result ─────────────────────────────────────


@dataclass
class PdfText:
    """Text extracted from a PDF, plus provenance for logging/debugging."""

    pages: List[str] = field(default_factory=list)
    engine: str = "none"
    tables_text: str = ""

    @property
    def text(self) -> str:
        """All pages joined with page markers (helps the LLM keep context)."""
        return "\n\n".join(
            f"--- PAGE {i + 1} ---\n{page}" for i, page in enumerate(self.pages)
        )

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)

    @property
    def is_usable(self) -> bool:
        """Enough real text that we can skip the vision (OCR) path."""
        return self.char_count > 50


# ─────────────────────────── Engine: pypdfium2 ──────────────────────────────


def _extract_pypdfium2(path: Path) -> List[str]:
    import pypdfium2 as pdfium

    pages: List[str] = []
    doc = pdfium.PdfDocument(str(path))
    try:
        for page in doc:
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_range().replace("\r\n", "\n"))
            finally:
                textpage.close()
            page.close()
    finally:
        doc.close()
    return pages


# ─────────────────────────── Engine: pdfplumber ─────────────────────────────


def _extract_pdfplumber(path: Path) -> tuple[List[str], str]:
    import pdfplumber

    pages: List[str] = []
    tables_text = ""
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
            for table in page.extract_tables():
                for row in table:
                    if not row:
                        continue
                    cells = [str(c).strip() if c else "" for c in row]
                    if any(cells):
                        tables_text += " | ".join(cells) + "\n"
    return pages, tables_text


# ──────────────────────────── Public API ────────────────────────────────────


def extract_pdf_text(path: Path) -> PdfText:
    """
    Extract text from *path* using the best available engine.

    Returns a :class:`PdfText`. An empty ``pages`` list means the PDF is a scan
    and the caller should fall back to vision/OCR.
    """
    fitz_pages: List[str] = []
    plumber_pages: List[str] = []
    tables_text = ""

    try:
        fitz_pages = _extract_pypdfium2(path)
    except Exception as exc:  # pragma: no cover - engine-specific failure
        logger.warning("pypdfium2 extraction failed: %s", exc)

    try:
        plumber_pages, tables_text = _extract_pdfplumber(path)
    except Exception as exc:  # pragma: no cover - engine-specific failure
        logger.warning("pdfplumber extraction failed: %s", exc)

    candidates = [
        ("pypdfium2", fitz_pages),
        ("pdfplumber", plumber_pages),
    ]

    best_engine, best_pages, best_score = "none", [], -1.0
    for engine, pages in candidates:
        joined = "\n".join(pages)
        if not joined.strip():
            continue
        # Prefer clean text; length only breaks ties between comparably clean
        # engines, so a long but corrupted extraction never wins.
        score = (1.0 - _mojibake_ratio(joined)) * 1000 + min(len(joined), 50_000) / 50_000
        logger.debug("PDF engine %s: %d chars, score %.3f", engine, len(joined), score)
        if score > best_score:
            best_engine, best_pages, best_score = engine, pages, score

    repaired = [repair_mojibake(p) for p in best_pages]

    if best_engine != "none":
        logger.info(
            "PDF text via %s — %d pages, %d chars (mojibake %.2f%%)",
            best_engine,
            len(repaired),
            sum(len(p) for p in repaired),
            _mojibake_ratio("\n".join(best_pages)) * 100,
        )

    return PdfText(
        pages=repaired,
        engine=best_engine,
        tables_text=repair_mojibake(tables_text),
    )


def render_pdf_pages(path: Path, scale: float = 2.0, max_pages: int = 10):
    """
    Render PDF pages to PIL images for the vision fallback.

    ``scale=2.0`` ≈ 144 DPI, which keeps small lab-report type legible without
    producing payloads that blow the model's image budget.
    """
    import pypdfium2 as pdfium

    images = []
    doc = pdfium.PdfDocument(str(path))
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                logger.warning("PDF has >%d pages — vision limited to the first %d", max_pages, max_pages)
                break
            images.append(page.render(scale=scale).to_pil())
            page.close()
    finally:
        doc.close()
    return images
