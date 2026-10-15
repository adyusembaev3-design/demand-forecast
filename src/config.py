"""Central configuration: paths and environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_PATH = DATA_PROCESSED / "forecast.duckdb"


def get_env(name: str, required: bool = True, default: str | None = None) -> str | None:
    """Read an environment variable with a helpful error if it is missing."""
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(
            f"Missing environment variable {name}. Copy .env.example to .env and fill it in."
        )
    return value
