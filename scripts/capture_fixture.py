"""
Capture Agent 1 + Agent 2 output for a real report as a test fixture.

Re-run this when the ontology or the extractor changes in a way that should
change the fixture. Committing the output keeps the reasoning tests offline.

Usage:
    python scripts/capture_fixture.py "<path to report.pdf>" tests/fixtures/<name>.json
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.pipeline import pipeline  # noqa: E402


async def main(pdf_path: str, out_path: str) -> None:
    state = await pipeline.ainvoke({"file_path": pdf_path, "file_type": "pdf"})
    fixture = {
        "source_file": Path(pdf_path).name,
        "extraction_method": state.get("extraction_method"),
        "extraction_degraded": bool(state.get("extraction_degraded")),
        "patient_info": state.get("patient_info"),
        "report_notes": state.get("report_notes") or [],
        "lab_results": state.get("lab_results") or [],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"wrote {out_path}: {len(fixture['lab_results'])} rows, "
          f"method={fixture['extraction_method']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
