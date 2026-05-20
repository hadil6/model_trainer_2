"""Backend entry point.

Run with:
    uvicorn main:app --port 8000 --reload
"""
from api.app import create_app
from core.logging_config import setup_logging

setup_logging()
app = create_app()
