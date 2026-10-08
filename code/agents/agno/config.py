

import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

CODE_ROOT = Path(__file__).resolve().parent

_env_file = os.getenv("ML_ENV_FILE", ".env")
if _env_file:
    load_dotenv(_env_file)
else:
    load_dotenv()


TMP_DIR = CODE_ROOT / "tmp"

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:7999/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.8-27B-4bit")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "local-key")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")       # optional -> enables KB
EMBEDDER_BASE_URL = os.getenv("EMBEDDER_BASE_URL", DEFAULT_LLM_BASE_URL)