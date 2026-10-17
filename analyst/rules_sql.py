"""Deterministic question -> SQL baseline used when no LLM is configured.

It is deliberately limited (metric x group-by x simple filters x top-N). It exists so that the whole
project runs offline and so the evaluation harness has an honest *baseline* to compare an LLM against.
"""
from __future__ import annotations

import re

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ORD_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4}

# metric -> (view, aggregate expression, alias)
METRICS = {
    "revenue": ("v_sales", "ROUND(SUM(revenue), 2)", "revenue"),
    "aov": ("v_sales", "ROUND(SUM(revenue) / COUNT(DISTINCT order_id), 2)", "aov"),
    "units": ("v_sales", "SUM(quantity)", "units_sold"),
    "orders": ("v_orders", "COUNT(*)", "orders"),
    "customers": ("v_orders", "COUNT(DISTINCT customer_id)", "customers"),
    "delivery": ("v_orders", "ROUND(AVG(delivery_days), 2)", "avg_delivery_days"),
    "rating": ("v_orders", "ROUND(AVG(rating), 2)", "avg_rating"),
    "cancel": ("v_orders", "ROUND(AVG(CASE WHEN status = 'cancelled' THEN 1.0 ELSE 0 END), 4)", "cancellation_rate"),
}
GROUPS = {
    "region": ["region"], "category": ["category"], "month": ["month_start"], "quarter": ["year", "quarter"],
    "year": ["year"], "customer_type": ["customer_type"], "status": ["status"], "product": ["product_id", "category"],
}


def _metric(q: str) -> str | None:
    if re.search(r"cancel", q):
        if re.search(r"rate|percent|share|proportion|%", q):
            return "cancel"
        return "orders"  # "how many orders were cancelled" is a count with a status filter
    if re.search(r"deliver\w*\s+(time|days)|shipping time|days to deliver|delivery duration|how long.*deliver", q):
        return "delivery"
    if re.search(r"rating|review score|stars", q):
        return "rating"
    if re.search(r"average order value|\baov\b|basket value|average order size|avg order", q):
        return "aov"
    if re.search(r"units|items sold|quantity|pieces", q):
        return "units"
    if re.search(r"how many (unique |distinct |active )?customers|number of (unique |distinct |active )?customers|unique customers|distinct customers|active customers|customers (placed|ordered|who)", q) \
            and not re.search(r"\b(revenue|sales)\b", q):
        return "customers"
    if re.search(r"\borders?\b", q):
        return "orders"
    if re.search(r"revenue|sales|gmv|income|earn|turnover|selling", q):
        return "revenue"
    return None


def parse_to_sql(question: str, regions: list[str], categories: list[str]) -> tuple[str | None, str]:
    """Return (sql, note). sql is None when the question is outside the rule grammar."""
    q = question.lower().strip().rstrip("?.!")
    metric = _metric(q)
    if metric is None:
        return None, "could not identify a metric"
    view, agg, alias = METRICS[metric]

    # --- group by ---
    group = None
    m = re.search(r"\b(?:by|per|each|every|across|for each|which)\s+(?:product\s+)?(region|categor\w+|month|quarter|year|customer type|status|product\w*|item\w*)", q)
    if m:
        g = m.group(1)
        group = "category" if g.startswith("categor") else "product" if (g.startswith("product") or g.startswith("item")) else "customer_type" if g == "customer type" else g
    elif re.search(r"monthly", q):
        group = "month"
    elif re.search(r"quarterly", q):
        group = "quarter"
    elif re.search(r"yearly|annual", q):
        group = "year"
    elif re.search(r"new (vs|versus|and) repeat", q):
        group = "customer_type"
    if group is None:
        m = re.search(r"\b(?:top|bottom|best|worst|highest|lowest)\s+\w*\s*(region|categor\w+|month|quarter|year|product\w*|item\w*)", q)
        if m:
            g = m.group(1)
            group = "category" if g.startswith("categor") else "product" if (g.startswith("product") or g.startswith("item")) else g

    # --- top / bottom ---
    order, limit = None, None
    n = None
    m = re.search(r"\b(?:top|bottom|lowest|highest|best|worst)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
    if re.search(r"\b(top|highest|most|best|largest|biggest|greatest|maximum|slowest|longest)\b", q):
        order, limit = "DESC", n or 1
    if re.search(r"\b(bottom|lowest|least|worst|smallest|fewest|minimum|fastest|shortest|quickest)\b", q):
        order, limit = "ASC", n or 1
    if order and group is None:
        return None, "ranking question without an entity to rank"

    # --- filters ---
    where = []
    if m := re.search(r"\b(20\d\d)\b", q):
        where.append(f"year = {m.group(1)}")
    qm = re.search(r"\bq([1-4])\b", q) or re.search(r"\b(first|second|third|fourth)\s+quarter", q)
    if qm:
        val = qm.group(1)
        where.append(f"quarter = 'Q{ORD_Q.get(val, val)}'")
    if not qm and group != "month":
        mm = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", q)
        if mm:
            where.append(f"month = {MONTHS[mm.group(1)]}")
    for r in regions:
        if re.search(rf"\b{r.lower()}\b", q) and group != "region":
            where.append(f"region = '{r}'")
    cat_hit = None
    for c in categories:
        key = c.lower().split(" &")[0]
        if re.search(rf"\b{re.escape(key)}\b", q) and group != "category":
            cat_hit = c
            where.append(f"category = '{c}'")
    if re.search(r"\brepeat customers?\b", q) and group != "customer_type":
        where.append("customer_type = 'repeat'")
    if re.search(r"\bnew customers?\b", q) and group != "customer_type":
        where.append("customer_type = 'new'")
    if re.search(r"\bcancel\w*\b", q) and metric in ("orders", "customers"):
        where.append("status = 'cancelled'")
    if re.search(r"\bdelivered orders\b", q) and metric in ("orders", "customers"):
        where.append("status = 'delivered'")
    if re.search(r"\breturned orders\b", q) and metric in ("orders", "customers"):
        where.append("status = 'returned'")

    if (group == "category" or cat_hit) and view == "v_orders":
        return None, "category is only available at item level (v_sales); this metric is order-level"
    if group == "status" and view == "v_sales":
        return None, "status grouping needs order-level data"

    cols = GROUPS.get(group, [])
    select = ", ".join(cols + [f"{agg} AS {alias}"])
    sql = f"SELECT {select} FROM {view}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if cols:
        sql += " GROUP BY " + ", ".join(cols)
        if order:
            sql += f" ORDER BY {alias} {order} LIMIT {limit}"
        elif group in ("month", "quarter", "year"):
            sql += " ORDER BY " + ", ".join(cols)
        else:
            sql += f" ORDER BY {alias} DESC"
    return sql, f"rule-based: metric={metric} group={group} filters={len(where)}"
