from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Model Trainer"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    mcp_port: int = 8001
    artifacts_dir: str = "artifacts"

    max_upload_mb: int = 50
    sample_seed: int = 42
    pdf_extraction_timeout_seconds: int = 7200

    azure_api_key: str = ""
    azure_endpoint: str = ""
    azure_api_version: str = "2025-04-01-preview"
    model_name: str = "gpt-5.4-mini"
    deployment_name: str = "gpt-5.4-mini"
    llm_max_tokens: int = 8192
    llm_timeout_seconds: float = 300.0
    llm_max_retries: int = 3

    request_timeout_seconds: float = 7200.0

    embedding_model: str = "intfloat/multilingual-e5-base"
    gpu_vram_gb: int = 4
    gpu_name: str = "NVIDIA GeForce RTX 2050"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow extra .env keys (e.g. SMTP_*, HPO_*) that are read directly
        # by other modules via os.environ rather than declared on Settings.
        # Without this, adding any new var to .env crashes Settings() and
        # therefore the whole app.
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
