"""Central settings. Everything can be overridden with environment variables or a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env loader (KEY=VALUE per line) so we don't need python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    db_path: Path = ROOT / "data" / "shop.duckdb"
    reports_dir: Path = ROOT / "reports"
    model: str = os.getenv("ANALYST_MODEL", "claude-sonnet-5-5")
    api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    max_sql_retries: int = 3
    row_limit: int = 5000
    currency: str = os.getenv("ANALYST_CURRENCY", "$")


def get_settings() -> Settings:
    return Settings()
