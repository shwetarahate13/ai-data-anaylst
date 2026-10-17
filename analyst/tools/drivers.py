"""Root-cause tools: driver attribution, greedy drill-down, volume/price split, mechanism scan.

Input frames carry a `period` column: 'base' (comparison period), 'window' (period being explained)
or anything else (ignored). All maths is plain pandas so every number is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ITEM_DIMS = ["region", "category", "customer_type", "signup_cohort", "basket_size"]
ORDER_DIMS = ["region", "customer_type", "signup_cohort", "basket_size"]


def segment_table(df: pd.DataFrame, dim: str, value_col: str) -> pd.DataFrame:
    t = (df[df["period"].isin(["base", "window"])]
         .pivot_table(index=dim, columns="period", values=value_col, aggfunc="sum", fill_value=0.0))
    for c in ("base", "window"):
        if c not in t:
            t[c] = 0.0
    t = t[["base", "window"]].copy()
    t["delta"] = t["window"] - t["base"]
    t["pct"] = np.where(t["base"] != 0, t["delta"] / t["base"].replace(0, np.nan), np.nan)
    return t.reset_index().rename(columns={dim: "segment"}).sort_values("delta").reset_index(drop=True)


@dataclass
class DrillStep:
    dim: str
    segment: str
    base: float
    window: float
    delta: float
    pct: float
    share_of_parent: float  # share of the parent slice's movement (in the asked direction)
    share_of_total: float   # slice delta / net total delta
    excess: float = float('nan')  # over-concentration vs the segment's share of the base (0 = proportional, 1 = all of it)


@dataclass
class Attribution:
    base_total: float
    window_total: float
    delta: float
    pct: float
    direction: str
    dim_tables: dict[str, pd.DataFrame]
    path: list[DrillStep] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)
    slice_delta: float = 0.0
    rest_delta: float = 0.0

    @property
    def slice_label(self) -> str:
        return describe_filters(self.filters)


def segment_label(dim: str, seg: str) -> str:
    return {"region": f"the {seg} region", "customer_type": f"{seg} customers", "category": f"{seg} products",
            "signup_cohort": f"customers who signed up in {seg}", "basket_size": f"{seg} baskets"}.get(dim, f"{dim}={seg}")


def describe_filters(filters: dict[str, str]) -> str:
    if not filters:
        return "the whole business"
    parts = []
    ct = filters.get("customer_type")
    if ct:
        parts.append(f"{ct} customers")
    else:
        parts.append("customers")
    if "region" in filters:
        parts[-1] += f" in the {filters['region']} region"
    if "category" in filters:
        parts.append(f"buying {filters['category']}")
    if "signup_cohort" in filters:
        parts.append(f"who signed up in {filters['signup_cohort']}")
    if "basket_size" in filters:
        parts.append(f"with {filters['basket_size']} baskets")
    return " ".join(parts)


def attribute(df: pd.DataFrame, value_col: str, dims: list[str], direction: str = "down",
              max_depth: int = 3, min_parent_share: float = 0.5, min_total_share: float = 0.2,
              min_excess: float = 0.5) -> Attribution:
    """Explain the change between 'base' and 'window' and greedily drill into the dominant segment.

    At each level we look across all remaining dimensions for the segment that moves the most in the
    asked direction, and keep drilling while that segment (a) carries at least `min_parent_share` of the
    parent slice's gross movement, (b) over-indexes its share of the base by `min_excess` (so a merely
    large segment doesn't count) and (c) is material relative to the total net change. If nothing passes,
    the path stays empty - the honest answer is then "the change is broad-based".
    """
    g = df.groupby("period")[value_col].sum()
    b, w = float(g.get("base", 0.0)), float(g.get("window", 0.0))
    total = w - b
    sign = -1.0 if direction == "down" else 1.0
    tables = {d: segment_table(df, d, value_col) for d in dims}

    path: list[DrillStep] = []
    filters: dict[str, str] = {}
    cur = df
    remaining = list(dims)
    for _ in range(max_depth):
        best = None
        for d in remaining:
            t = segment_table(cur, d, value_col)
            moves = t["delta"] * sign
            gross = float(moves[moves > 0].sum())
            base_sum = float(t["base"].sum())
            if gross <= 0 or base_sum <= 0:
                continue
            i = int(moves.idxmax())
            share_move = float(moves.loc[i] / gross)
            share_base = float(t.loc[i, "base"] / base_sum)
            # A segment is only a "driver" if it carries MORE of the movement than its weight in the
            # base. excess = 0 -> proportional (just a big segment); 1 -> it is all of the movement.
            excess = (share_move - share_base) / (1 - share_base) if share_base < 0.98 else -1.0
            if best is None or excess > best[0]:
                best = (excess, share_move, d, t.loc[i])
        if best is None:
            break
        excess, share_move, d, top = best
        if excess < min_excess or share_move < min_parent_share or abs(top["delta"]) < min_total_share * abs(total):
            break
        path.append(DrillStep(d, str(top["segment"]), float(top["base"]), float(top["window"]), float(top["delta"]),
                              float(top["pct"]) if pd.notna(top["pct"]) else float("nan"), share_move,
                              float(top["delta"] / total) if total else float("nan"), float(excess)))
        filters[d] = str(top["segment"])
        cur = cur[cur[d] == top["segment"]]
        remaining.remove(d)

    slice_delta = path[-1].delta if path else total
    if filters:
        sl = df
        for k, v in filters.items():
            sl = sl[sl[k] == v]
        gs = sl.groupby("period")[value_col].sum()
        slice_delta = float(gs.get("window", 0.0) - gs.get("base", 0.0))
    return Attribution(b, w, total, (total / b if b else float("nan")), direction, tables, path, filters,
                       slice_delta, total - slice_delta)


def volume_price_split(orders_delivered: pd.DataFrame) -> dict:
    """Revenue = delivered orders x AOV. delta = volume effect + price/mix effect (exact, no residual)."""
    d = orders_delivered
    n0, n1 = int((d["period"] == "base").sum()), int((d["period"] == "window").sum())
    r0 = float(d.loc[d["period"] == "base", "order_revenue"].sum())
    r1 = float(d.loc[d["period"] == "window", "order_revenue"].sum())
    a0, a1 = (r0 / n0 if n0 else 0.0), (r1 / n1 if n1 else 0.0)
    return {"orders_base": n0, "orders_window": n1, "aov_base": a0, "aov_window": a1,
            "volume_effect": (n1 - n0) * a0, "price_effect": n1 * (a1 - a0)}


def _stats(x: pd.Series):
    x = x.dropna()
    n = len(x)
    return n, (float(x.mean()) if n else np.nan), (float(x.var(ddof=1)) if n > 1 else np.nan)


def mechanism_scan(orders: pd.DataFrame, filters: dict[str, str], z_threshold: float = 3.0) -> pd.DataFrame:
    """Which operational metric moved in the slice *differently from the rest of the business*?

    Difference-in-differences per metric: (slice_window - slice_base) - (rest_window - rest_base),
    standardised with a Welch-style standard error. Only order-level dimensions can be used to cut the slice.
    """
    f = {k: v for k, v in filters.items() if k in ORDER_DIMS}
    if not f:
        return pd.DataFrame()
    mask = pd.Series(True, index=orders.index)
    for k, v in f.items():
        mask &= orders[k] == v
    o = orders[orders["period"].isin(["base", "window"])].copy()
    o["cancelled"] = (o["status"] == "cancelled").astype(float)
    mask = mask.loc[o.index]
    rows = []
    for metric in ["delivery_days", "rating", "cancelled"]:
        cells = {}
        for name, m in (("slice", mask), ("rest", ~mask)):
            for per in ("base", "window"):
                cells[(name, per)] = _stats(o.loc[m & (o["period"] == per), metric])
        if any(c[0] < 2 for c in cells.values()):
            continue
        sb, sw, rb, rw = cells[("slice", "base")], cells[("slice", "window")], cells[("rest", "base")], cells[("rest", "window")]
        did = (sw[1] - sb[1]) - (rw[1] - rb[1])
        se = np.sqrt(sum(c[2] / c[0] for c in (sb, sw, rb, rw)))
        rows.append({"metric": metric, "slice_base": sb[1], "slice_window": sw[1], "slice_change": sw[1] - sb[1],
                     "rest_base": rb[1], "rest_window": rw[1], "rest_change": rw[1] - rb[1],
                     "did": did, "z": did / se if se > 0 else np.nan, "n_slice_window": sw[0]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant"] = out["z"].abs() >= z_threshold
        out = out.reindex(out["z"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out
