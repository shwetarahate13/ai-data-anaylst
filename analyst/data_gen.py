"""Synthetic e-commerce dataset with a *known, injected* root cause.

Why synthetic? It lets us measure the agent against ground truth. The injected story:

    From 2024-07-01 to 2024-09-30, deliveries to the WEST region slowed by ~3.3 days.
    REPEAT customers in the West reacted by ordering ~65% less. Nothing else changed
    structurally (plus two small red herrings: a Fashion price promo in Aug 2024 and
    normal seasonality / growth).

A good analyst agent should discover: "revenue fell in Q3 2024, mostly because repeat customers in
the West ordered less, coinciding with a ~3-day jump in West delivery time".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_P = [0.20, 0.22, 0.18, 0.25, 0.15]
CATEGORIES = {  # name: (weight, min price, max price, n products)
    "Electronics": (0.12, 80, 600, 18),
    "Home & Kitchen": (0.22, 15, 120, 22),
    "Fashion": (0.25, 10, 90, 24),
    "Beauty": (0.12, 8, 60, 14),
    "Sports": (0.10, 15, 150, 14),
    "Books": (0.10, 5, 30, 14),
    "Toys": (0.09, 8, 70, 14),
}
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
SHOCK_START, SHOCK_END = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30")
SHOCK_REGION = "West"
SHOCK_DELIVERY_DELTA = 3.3
SHOCK_REPEAT_KEEP = 0.35  # fraction of West repeat orders that still happen during the shock
SEASON = [0.92, 0.90, 0.97, 1.00, 1.02, 1.00, 0.98, 1.00, 1.02, 1.08, 1.30, 1.35]


def generate(seed: int = 7, n_customers: int = 14000, base_daily_orders: float = 75.0):
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")

    # --- customers (sorted by signup so "active so far" is a prefix) ---
    offsets = (rng.beta(1.2, 1.8, size=n_customers) * 1094).astype(int) - 365
    signup = (START + pd.to_timedelta(np.sort(offsets), unit="D")).normalize()
    region = rng.choice(REGIONS, size=n_customers, p=REGION_P)
    customers = pd.DataFrame({"customer_id": np.arange(1, n_customers + 1), "region": region, "signup_date": signup})

    # --- products ---
    rows = []
    pid = 1
    for cat, (_, lo, hi, n) in CATEGORIES.items():
        prices = np.round(np.exp(rng.uniform(np.log(lo), np.log(hi), size=n)), 2)
        for k, price in enumerate(prices, 1):
            rows.append((pid, f"{cat} item {k}", cat, float(price)))
            pid += 1
    products = pd.DataFrame(rows, columns=["product_id", "product_name", "category", "price"])

    # --- orders, simulated day by day ---
    seen = np.zeros(n_customers, dtype=bool)
    signup_arr = customers["signup_date"].to_numpy()
    region_arr = customers["region"].to_numpy()
    o_date, o_cust = [], []
    for i, d in enumerate(days):
        lam = base_daily_orders * (1 + 0.35 * i / len(days)) * SEASON[d.month - 1] * (1.10 if d.dayofweek >= 5 else 1.0)
        n = rng.poisson(lam)
        active = int(np.searchsorted(signup_arr, np.datetime64(d), side="right"))
        if n == 0 or active == 0:
            continue
        cust = rng.integers(0, active, size=n)
        if SHOCK_START <= d <= SHOCK_END:
            drop = (region_arr[cust] == SHOCK_REGION) & seen[cust] & (rng.random(n) > SHOCK_REPEAT_KEEP)
            cust = cust[~drop]
        seen[cust] = True
        o_date.extend([d] * len(cust))
        o_cust.extend(cust + 1)
    orders = pd.DataFrame({"order_id": np.arange(1, len(o_date) + 1), "customer_id": o_cust, "order_date": o_date})

    reg = region_arr[orders["customer_id"].to_numpy() - 1]
    in_shock = ((orders["order_date"] >= SHOCK_START) & (orders["order_date"] <= SHOCK_END)).to_numpy()
    delivery = np.clip(rng.normal(4.2, 1.1, len(orders)), 1, 14)
    delivery = delivery + np.where(in_shock & (reg == SHOCK_REGION), SHOCK_DELIVERY_DELTA, 0.0)
    status = rng.choice(["delivered", "cancelled", "returned"], size=len(orders), p=[0.92, 0.05, 0.03])
    orders["status"] = status
    orders["delivery_days"] = np.where(status == "cancelled", np.nan, np.round(delivery, 1))

    # --- items ---
    n_items = rng.choice([1, 2, 3], size=len(orders), p=[0.55, 0.30, 0.15])
    item_order = np.repeat(orders["order_id"].to_numpy(), n_items)
    item_date = np.repeat(orders["order_date"].to_numpy(), n_items)
    cats = list(CATEGORIES)
    cat_pick = rng.choice(cats, size=len(item_order), p=[CATEGORIES[c][0] for c in cats])
    by_cat = {c: products[products.category == c] for c in cats}
    pid_pick = np.zeros(len(item_order), dtype=int)
    price_pick = np.zeros(len(item_order))
    for c in cats:
        m = cat_pick == c
        sub = by_cat[c].reset_index(drop=True)
        idx = rng.integers(0, len(sub), size=m.sum())
        pid_pick[m] = sub["product_id"].to_numpy()[idx]
        price_pick[m] = sub["price"].to_numpy()[idx]
    promo = (cat_pick == "Fashion") & (item_date >= np.datetime64("2024-08-01")) & (item_date <= np.datetime64("2024-08-31"))
    price_pick = np.round(price_pick * np.where(promo, 0.92, 1.0), 2)  # red herring: small Fashion promo
    qty = rng.choice([1, 2, 3], size=len(item_order), p=[0.80, 0.15, 0.05])
    order_items = pd.DataFrame({
        "order_item_id": np.arange(1, len(item_order) + 1), "order_id": item_order,
        "product_id": pid_pick, "quantity": qty, "unit_price": price_pick,
    })

    # --- reviews (rating drops when delivery is slow) ---
    deliv = orders[orders.status == "delivered"].copy()
    deliv = deliv[rng.random(len(deliv)) < 0.70]
    rating = np.clip(np.round(4.4 - 0.30 * (deliv["delivery_days"] - 4.2) + rng.normal(0, 0.8, len(deliv))), 1, 5).astype(int)
    reviews = pd.DataFrame({"order_id": deliv["order_id"].to_numpy(), "rating": rating.to_numpy()})

    truth = {
        "shock_window": [str(SHOCK_START.date()), str(SHOCK_END.date())],
        "region": SHOCK_REGION,
        "customer_type": "repeat",
        "mechanism": "delivery_days",
        "delivery_delta_days": SHOCK_DELIVERY_DELTA,
        "repeat_keep_probability": SHOCK_REPEAT_KEEP,
        "red_herrings": ["Fashion price promo Aug 2024 (-8%)", "seasonality", "organic growth"],
    }
    return {"customers": customers, "products": products, "orders": orders, "order_items": order_items, "reviews": reviews}, truth


def write_csvs(out_dir: Path, **kwargs) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, truth = generate(**kwargs)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return {k: len(v) for k, v in frames.items()}
