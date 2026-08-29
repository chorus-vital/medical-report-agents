"""
Application settings loaded from environment variables / .env file.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pydantic Settings for the Medical Report Analyzer."""

    # --- AI Model Provider ---
    MODEL_PROVIDER: Literal["gemini", "ollama", "groq"] = "gemini"

    # --- Google Gemini ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    # --- Ollama ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"

    # --- Groq ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Application ---
    SQLITE_DB_PATH: str = "data/medical_reports.db"
    UPLOAD_DIR: str = "data/uploads"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LAB_ONTOLOGY_PATH: str = "data/lab_ontology.json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
