"""
Unified LLM factory — seamlessly switch between Gemini, Ollama, and Groq.

Usage:
    from src.services.llm_factory import get_chat_model, get_vision_model

    llm = get_chat_model()              # LangChain ChatModel for text tasks
    vision = get_vision_model()          # google-genai client for images
"""

import logging

from google import genai
from langchain_core.language_models import BaseChatModel

from config.settings import settings

logger = logging.getLogger(__name__)


def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """
    Return a LangChain ChatModel based on the MODEL_PROVIDER setting.

    Providers:
      - ``gemini``  → Google Gemini Flash (free AI Studio quota)
      - ``ollama``  → Local Ollama (100 % offline, zero cost)
      - ``groq``    → Groq cloud (free daily quota, ultra-fast)
    """
    provider = settings.MODEL_PROVIDER.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required when MODEL_PROVIDER=gemini. "
                "Get a free key at https://aistudio.google.com"
            )
        logger.info("Using Gemini model: %s", settings.GEMINI_MODEL)
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info("Using Ollama model: %s @ %s", settings.OLLAMA_MODEL, settings.OLLAMA_BASE_URL)
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is required when MODEL_PROVIDER=groq. "
                "Get a free key at https://console.groq.com"
            )
        logger.info("Using Groq model: %s", settings.GROQ_MODEL)
        return ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown MODEL_PROVIDER: '{provider}'. "
        "Choose from: 'gemini', 'ollama', 'groq'."
    )


def get_vision_model() -> genai.Client:
    """
    Return a ``google.genai.Client`` for multimodal (vision) requests.

    Always uses Gemini regardless of ``MODEL_PROVIDER`` because local
    Ollama / Groq don't expose a comparable free-tier vision API.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is not configured.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is required for vision / image extraction. "
            "Get a free key at https://aistudio.google.com"
        )
    logger.info("Using Gemini Vision model: %s", settings.GEMINI_MODEL)
    return genai.Client(api_key=settings.GEMINI_API_KEY)
