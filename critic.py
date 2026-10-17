"""Critic agent: independently verifies every number before it reaches the narrator.

Strategy: the analysis path uses the analytic views + pandas. The Critic recomputes the headline
figures from the RAW tables with separately written SQL, checks internal consistency (segments must
add up to the total) and statistical sufficiency (sample size, significance). Any 'fail' triggers a
recompute in the orchestrator; whatever is still failing lowers the reported confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .db import Database
from .utils import sql_date

DIM_EXPR = {
    "region": "c.region",
    "category": "p.category",
    "customer_type": "CASE WHEN ord.rn = 1 THEN 'new' ELSE 'repeat' END",
    "signup_cohort": "CAST(EXTRACT(year FROM c.signup_date) AS VARCHAR)",
    "basket_size": "CASE WHEN ic.n = 1 THEN '1 item' WHEN ic.n = 2 THEN '2 items' ELSE '3+ items' END",
}


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail | info
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def raw_total_sql(metric: str, start: pd.Timestamp, end: pd.Timestamp, filters: dict[str, str]) -> str:
    """Total for a slice computed from base tables only (never touches v_orders / v_sales)."""
    conds = [f"ord.order_date >= {sql_date(start)}", f"ord.order_date < {sql_date(end)}"]
    for k, v in filters.items():
        conds.append(f"{DIM_EXPR[k]} = '{str(v).replace(chr(39), chr(39) * 2)}'")
    if metric == "revenue":
        conds.append("ord.status = 'delivered'")
        select, items = "COALESCE(SUM(oi.quantity * oi.unit_price), 0)", "JOIN order_items oi ON oi.order_id = ord.order_id JOIN products p ON p.product_id = oi.product_id"
    else:
        select, items = "COUNT(DISTINCT ord.order_id)", "JOIN order_items oi ON oi.order_id = ord.order_id JOIN products p ON p.product_id = oi.product_id"
    return (
        "WITH ord AS (SELECT order_id, customer_id, order_date, status, "
        "ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date, order_id) AS rn FROM orders), "
        "ic AS (SELECT order_id, COUNT(*) AS n FROM order_items GROUP BY order_id) "
        f"SELECT {select} AS total FROM ord JOIN customers c ON c.customer_id = ord.customer_id "
        f"JOIN ic ON ic.order_id = ord.order_id {items} WHERE " + " AND ".join(conds)
    )


def _close(a: float, b: float, rel: float = 1e-6, abs_tol: float = 0.01) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


class Critic:
    def __init__(self, db: Database):
        self.db = db

    def _raw(self, metric, start, end, filters) -> float:
        return float(self.db.pull(raw_total_sql(metric, start, end, filters)).iloc[0, 0])

    def verify(self, *, metric: str, window, baseline, attr, monthly: pd.Series, slice_counts: dict | None,
               mechanism: pd.DataFrame | None, anomaly_window) -> list[Check]:
        checks: list[Check] = []
        ws, we = window
        bs, be = baseline

        # 1. headline totals vs the independent monthly series
        m_w = float(monthly[(monthly.index >= ws) & (monthly.index < we)].sum())
        m_b = float(monthly[(monthly.index >= bs) & (monthly.index < be)].sum())
        ok = _close(m_w, attr.window_total) and _close(m_b, attr.base_total)
        checks.append(Check("Totals match monthly series", "pass" if ok else "fail",
                            f"window {attr.window_total:,.2f} vs {m_w:,.2f}; baseline {attr.base_total:,.2f} vs {m_b:,.2f}"))

        # 2. headline totals vs raw-table SQL (independent code path)
        raw_w, raw_b = self._raw(metric, ws, we, {}), self._raw(metric, bs, be, {})
        ok = _close(raw_w, attr.window_total) and _close(raw_b, attr.base_total)
        checks.append(Check("Totals match raw-table SQL", "pass" if ok else "fail",
                            f"window {attr.window_total:,.2f} vs {raw_w:,.2f}; baseline {attr.base_total:,.2f} vs {raw_b:,.2f}"))

        # 3. every dimension must reconcile to the total change
        bad = [d for d, t in attr.dim_tables.items() if not _close(float(t["delta"].sum()), attr.delta)]
        checks.append(Check("Segments add up to total change", "pass" if not bad else "fail",
                            "all dimensions reconcile" if not bad else f"mismatch in: {', '.join(bad)}"))

        # 4. the headline slice, re-derived from raw tables
        if attr.filters:
            rw, rb = self._raw(metric, ws, we, attr.filters), self._raw(metric, bs, be, attr.filters)
            # slice delta from the attribution frame
            ok = _close(rw - rb, attr.slice_delta)
            checks.append(Check("Top-driver slice matches raw-table SQL", "pass" if ok else "fail",
                                f"slice change {attr.slice_delta:,.2f} vs raw {rw - rb:,.2f}"))

        # 5. enough data behind the claim?
        if slice_counts:
            n_b, n_w = slice_counts["base"], slice_counts["window"]
            ok = n_b >= 30 and n_w >= 30
            checks.append(Check("Sample size", "pass" if ok else "warn", f"{n_b} orders in baseline slice, {n_w} in window slice"))

        # 6. is the operational mechanism statistically distinguishable from noise?
        if mechanism is not None and not mechanism.empty:
            top = mechanism.iloc[0]
            ok = bool(top["significant"])
            checks.append(Check("Mechanism is statistically strong", "pass" if ok else "warn",
                                f"{top['metric']}: difference-in-differences {top['did']:+.2f}, z = {top['z']:.1f}"))

        # 7. does the anomaly detector independently see the same period?
        if anomaly_window is not None:
            checks.append(Check("Anomaly detector corroborates the period", "pass",
                                f"detector flagged {anomaly_window.start:%b %Y}-{(anomaly_window.end - pd.DateOffset(months=1)):%b %Y} "
                                f"(t = {anomaly_window.severity:.1f}; Isolation Forest agrees on {anomaly_window.corroborated:.0%} of months)"))
        else:
            checks.append(Check("Anomaly detector corroborates the period", "info",
                                "detector did not flag this exact period as unusual on a seasonally-adjusted basis"))

        checks.append(Check("Causal caution", "info", "Observational data: findings show association, not proven causation."))
        return checks

    @staticmethod
    def confidence(checks: list[Check]) -> str:
        if any(c.status == "fail" for c in checks):
            return "low"
        if any(c.status == "warn" for c in checks):
            return "medium"
        return "high"
