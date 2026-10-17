"""What-if simulation: how would repeat orders respond if delivery got faster?

Model (two-way fixed effects on a region x week panel):
    log(1 + repeat_orders[r, w]) = region_r + week_w + beta * avg_delivery_days[r, w] + noise
The week effects absorb company-wide seasonality/trend and the region effects absorb size, so beta is
identified from region-specific delivery changes (e.g. the West slowdown). It is an observational
elasticity, not a randomised experiment - the output says so, and flags extrapolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_panel(orders: pd.DataFrame) -> pd.DataFrame:
    o = orders.copy()
    o["week"] = pd.to_datetime(o["order_date"]).dt.to_period("W").dt.start_time
    rep = o[o["customer_type"] == "repeat"].groupby(["region", "week"]).size().rename("repeat_orders")
    days = o.groupby(["region", "week"])["delivery_days"].mean().rename("avg_days")
    p = pd.concat([rep, days], axis=1).dropna().reset_index()
    # drop first/last partial weeks
    p = p[(p["week"] > p["week"].min()) & (p["week"] < p["week"].max())]
    return p


def fit_elasticity(panel: pd.DataFrame) -> dict:
    y = np.log1p(panel["repeat_orders"].to_numpy(dtype=float))
    X = pd.concat([
        pd.get_dummies(panel["region"], prefix="r", drop_first=True),
        pd.get_dummies(panel["week"].astype(str), prefix="w", drop_first=True),
    ], axis=1).astype(float)
    X.insert(0, "const", 1.0)
    X["days"] = panel["avg_days"].to_numpy(dtype=float)
    A = X.to_numpy()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(y) - A.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    j = A.shape[1] - 1
    beta, se = float(coef[j]), float(np.sqrt(cov[j, j]))
    return {"beta": beta, "se": se, "t": beta / se if se else float("nan"), "n": int(len(y)),
            "r2": float(1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())))}


def simulate(orders: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp], reduction: float = 0.20,
             region: str | None = None) -> dict:
    """Project extra repeat-customer revenue if delivery time fell by `reduction` in the window."""
    panel = build_panel(orders)
    fit = fit_elasticity(panel)
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    scope = panel[(panel["week"] >= start) & (panel["week"] < end)]
    o = orders[(orders["order_date"] >= start) & (orders["order_date"] < end)]
    if region:
        scope = scope[scope["region"] == region]
        o = o[o["region"] == region]
    rep = o[o["customer_type"] == "repeat"]
    delivered_rate = float((rep["status"] == "delivered").mean()) if len(rep) else 0.0
    aov = float(rep.loc[rep["status"] == "delivered", "order_revenue"].mean()) if (rep["status"] == "delivered").any() else 0.0

    delta_days = -reduction * scope["avg_days"].to_numpy()

    def uplift(beta: float) -> tuple[float, float]:
        extra = scope["repeat_orders"].to_numpy() * (np.exp(beta * delta_days) - 1.0)
        return float(extra.sum()), float(extra.sum() * delivered_rate * aov)

    orders_mid, rev_mid = uplift(fit["beta"])
    lo = uplift(fit["beta"] + 1.96 * fit["se"])[1]
    hi = uplift(fit["beta"] - 1.96 * fit["se"])[1]
    q05 = float(panel["avg_days"].quantile(0.05))
    new_days = float((scope["avg_days"] * (1 - reduction)).mean()) if len(scope) else float("nan")
    return {
        "reduction": reduction, "region": region or "all regions", "beta": fit["beta"], "beta_se": fit["se"], "beta_t": fit["t"],
        "panel_rows": fit["n"], "avg_days_before": float(scope["avg_days"].mean()) if len(scope) else float("nan"),
        "avg_days_after": new_days, "extra_repeat_orders": orders_mid, "extra_revenue": rev_mid,
        "revenue_low": min(lo, hi), "revenue_high": max(lo, hi),
        "extrapolation": bool(new_days < q05),
        "caveat": "Observational elasticity from historical region-week variation; treat as an estimate, not a guarantee.",
    }
