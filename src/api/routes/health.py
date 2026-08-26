"""
Health-check & model-status endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter

from config.settings import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Return service health and the active AI model provider."""

    provider = settings.MODEL_PROVIDER.lower()
    model_map = {
        "gemini": settings.GEMINI_MODEL,
        "ollama": settings.OLLAMA_MODEL,
        "groq": settings.GROQ_MODEL,
    }

    return {
        "status": "healthy",
        "model_provider": provider,
        "model_name": model_map.get(provider, "unknown"),
    }
