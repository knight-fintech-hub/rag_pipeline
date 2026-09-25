"""
Root-level application configuration.
Reads settings from environment variables / .env files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env.local first (developer overrides), then .env (defaults)
_BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE, ".env.local"), override=True)
load_dotenv(os.path.join(_BASE, ".env"), override=False)


def _parse_bool(key: str, default: bool) -> bool:
    """Parse an env variable as a boolean. Accepts true/1/yes (case-insensitive)."""
    val = os.environ.get(key, "")
    if val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes")


@dataclass
class Config:
    # PostgreSQL DSN — e.g. "postgresql://user:pass@host:5432/dbname"
    DATABASE_URL: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL", "")
    )

    # When True, the app connects to PostgreSQL for session history.
    # Set USE_DB=false in .env to run fully without a database.
    USE_DB: bool = field(
        default_factory=lambda: _parse_bool("USE_DB", True)
    )

    # When True, retrieved chunks are written to session_chunks/<session_id>.json
    # for debugging. Set DEBUG=true in .env to enable.
    DEBUG: bool = field(
        default_factory=lambda: _parse_bool("DEBUG", False)
    )


cfg = Config()
