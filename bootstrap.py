"""Create the demo dataset + DuckDB file on first use."""
from __future__ import annotations

from pathlib import Path

from .config import Settings, get_settings
from .data_gen import write_csvs
from .db import build_database


def ensure_database(settings: Settings | None = None, force: bool = False) -> Path:
    settings = settings or get_settings()
    if force or not settings.db_path.exists():
        counts = write_csvs(settings.data_dir)
        build_database(settings.data_dir, settings.db_path)
        print(f"[setup] generated demo data {counts} -> {settings.db_path}")
    return settings.db_path
