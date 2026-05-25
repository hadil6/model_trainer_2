"""Backend entry point.

Run with:
    uvicorn main:app --port 8000 --reload
"""
# Load .env into os.environ BEFORE anything else imports modules that read
# environment variables (email_notifier, swift_runner, etc.). pydantic's
# Settings does its own .env loading but stays within its schema — modules
# that read os.environ directly need this explicit load.
from dotenv import load_dotenv
load_dotenv()

from api.app import create_app
from core.logging_config import setup_logging

setup_logging()
app = create_app()
