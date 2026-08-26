"""
Compiled LangGraph StateGraph with streaming support.

The pipeline chains three nodes sequentially:

    extract  →  ground  →  reason_and_verify  →  END

Currently only ``extract`` is fully implemented; the other two are
pass-through placeholders.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.graph.nodes import extract_node, ground_node, reason_and_verify_node
from src.schemas.state import PipelineState

logger = logging.getLogger(__name__)


def build_pipeline() -> StateGraph:
    """Construct and compile the 3-step analysis graph."""

    graph = StateGraph(PipelineState)

    # ── register nodes ──
    graph.add_node("extract", extract_node)
    graph.add_node("ground", ground_node)
    graph.add_node("reason_and_verify", reason_and_verify_node)

    # ── linear flow ──
    graph.set_entry_point("extract")
    graph.add_edge("extract", "ground")
    graph.add_edge("ground", "reason_and_verify")
    graph.add_edge("reason_and_verify", END)

    compiled = graph.compile()
    logger.info("✅ Pipeline compiled (extract → ground → reason_and_verify → END)")
    return compiled


# Module-level singleton so the rest of the app can import it directly.
pipeline = build_pipeline()
