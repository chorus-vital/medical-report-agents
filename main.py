"""
Medical Report Analyzer — Single Entrypoint
Run: python main.py
Server: http://localhost:8000 (API docs at http://localhost:8000/docs)
"""

import uvicorn

from config.settings import settings
# Vercel loads the ASGI app from this root entrypoint.
from src.api.app import app  # noqa: F401


def main():
    """Start the FastAPI application server."""
    uvicorn.run(
        "src.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
