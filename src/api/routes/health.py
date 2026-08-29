"""
Health-check & model-status endpoints.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# A live probe costs a network round-trip, so the result is cached briefly.
_PROBE_CACHE: dict[str, object] = {}
_PROBE_TTL_SECONDS = 60.0


def _active_model() -> str:
    return {
        "gemini": settings.GEMINI_MODEL,
        "ollama": settings.OLLAMA_MODEL,
        "groq": settings.GROQ_MODEL,
    }.get(settings.MODEL_PROVIDER.lower(), "unknown")


@router.get("/health")
async def health_check():
    """Return service health and the configured AI model provider."""
    provider = settings.MODEL_PROVIDER.lower()
    configured = provider != "gemini" or bool(settings.GEMINI_API_KEY)

    return {
        "status": "healthy",
        "model_provider": provider,
        "model_name": _active_model(),
        "api_key_configured": configured,
    }


@router.get("/health/llm")
async def llm_health_check():
    """
    Verify the configured model actually answers.

    ``/health`` only reports configuration; a missing or rejected API key still
    looks "healthy" there while every extraction silently degrades to the regex
    fallback. This endpoint makes that failure visible before a user uploads
    anything.
    """
    provider = settings.MODEL_PROVIDER.lower()
    model = _active_model()

    cached = _PROBE_CACHE.get("result")
    cached_at = _PROBE_CACHE.get("at")
    now = asyncio.get_event_loop().time()
    if cached and isinstance(cached_at, float) and now - cached_at < _PROBE_TTL_SECONDS:
        return {**cached, "cached": True}  # type: ignore[dict-item]

    if provider == "gemini" and not settings.GEMINI_API_KEY:
        result = {
            "status": "misconfigured",
            "model_provider": provider,
            "model_name": model,
            "reachable": False,
            "detail": (
                "GEMINI_API_KEY is not set. Extraction will fall back to a basic "
                "pattern matcher and miss most results. Add the key to your .env "
                "file and restart the server."
            ),
        }
        _PROBE_CACHE.update(result=result, at=now)
        return {**result, "cached": False}

    try:
        detail = await _probe(provider)
        result = {
            "status": "healthy",
            "model_provider": provider,
            "model_name": model,
            "reachable": True,
            "detail": detail,
        }
    except Exception as exc:
        logger.error("LLM health probe failed: %s", exc)
        result = {
            "status": "unreachable",
            "model_provider": provider,
            "model_name": model,
            "reachable": False,
            "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
        }

    _PROBE_CACHE.update(result=result, at=now)
    return {**result, "cached": False}


async def _probe(provider: str) -> str:
    """Send the smallest possible request the provider will accept."""
    if provider == "gemini":
        from src.services.llm_factory import get_vision_model

        client = get_vision_model()
        response = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents="Reply with the single word: ok",
        )
        return f"Model replied: {(response.text or '').strip()[:50]}"

    from src.services.llm_factory import get_chat_model

    llm = get_chat_model()
    response = await llm.ainvoke("Reply with the single word: ok")
    content = getattr(response, "content", response)
    return f"Model replied: {str(content).strip()[:50]}"
