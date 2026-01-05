"""Centralized configuration and environment variable management."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Find project root by looking for .env file
def _find_project_root() -> Path:
    """Find the project root by traversing up to find .env file."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".env").exists():
            return parent
    # Fallback to src parent
    return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = _find_project_root()

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# LLM Provider API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# IBM WatsonX
IBM_WATSON_PROJECT_ID = os.getenv("IBM_WATSON_PROJECT_ID")
IBM_WATSONX_APIKEY = os.getenv("IBM_WATSONX_APIKEY")
IBM_WATSONX_URL = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with optional default."""
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """Get required environment variable, raise if not set."""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

