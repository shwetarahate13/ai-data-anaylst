"""Shared helpers: agent trace, number formatting, date windows."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import pandas as pd


@dataclass
class TraceEvent:
    agent: str
    action: str
    detail: str = ""
    sql: str | None = None
    t: float = field(default_factory=time.time)


class Trace:
    """Ordered log of everything the agents did. Shown in the UI so the work is inspectable."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def log(self, agent: str, action: str, detail: str = "", sql: str | None = None) -> None:
        self.events.append(TraceEvent(agent, action, detail, sql))

    def to_list(self) -> list[dict]:
        t0 = self.events[0].t if self.events else 0
        return [{**asdict(e), "t": round(e.t - t0, 2)} for e in self.events]


def money(x: float, cur: str = "$") -> str:
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e6:
        return f"{sign}{cur}{a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}{cur}{a / 1e3:.1f}K"
    return f"{sign}{cur}{a:,.0f}"


def pct(x: float, signed: bool = False, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def quarter_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Human label for a month-aligned [start, end) window."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if months == 3 and start.month in (1, 4, 7, 10):
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    if months == 1:
        return start.strftime("%b %Y")
    last = end - pd.DateOffset(months=1)
    return f"{start.strftime('%b %Y')} to {last.strftime('%b %Y')}"


def sql_date(ts: pd.Timestamp) -> str:
    return f"DATE '{pd.Timestamp(ts).strftime('%Y-%m-%d')}'"
