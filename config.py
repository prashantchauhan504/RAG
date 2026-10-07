import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()


def get(key: str, default: str = "") -> str:
    """Read a string from environment. Crashes loudly if a required key is missing."""
    value = os.getenv(key, default)
    return value


def get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def require(key: str) -> str:
    """Read a required key. Raises a clear error if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Add it to your .env file."
        )
    return value


# --- Settings used across the project ---

GOOGLE_API_KEY    = require("GOOGLE_API_KEY")
PINECONE_API_KEY  = require("PINECONE_API_KEY")
PINECONE_INDEX    = get("PINECONE_INDEX_NAME", "rag-index")

CHUNK_SIZE        = int(get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP     = int(get("CHUNK_OVERLAP", "50"))

# Query-time settings (Phase B)
TOP_K             = int(get("TOP_K", "5"))
GOOGLE_LLM_MODEL  = get("GOOGLE_LLM_MODEL", "gemini-1.5-flash")

# Shared embedding model for both ingest + query
EMBEDDING_MODEL   = get("EMBEDDING_MODEL", "models/text-embedding-004")
ALLOW_INDEX_RECREATE = get_bool("ALLOW_INDEX_RECREATE", True)
