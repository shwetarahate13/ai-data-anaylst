"""Schema documentation + business glossary. These become the chunks retrieved by schema RAG."""
from __future__ import annotations

import json
from pathlib import Path

GLOSSARY: dict[str, str] = json.loads((Path(__file__).parent / "glossary.json").read_text())

SCHEMA_DOCS: list[dict] = [
    {"name": "customers", "text": "TABLE customers(customer_id INTEGER, region VARCHAR, signup_date DATE). One row per customer. region is North, South, East, West or Central."},
    {"name": "products", "text": "TABLE products(product_id INTEGER, product_name VARCHAR, category VARCHAR, price DOUBLE). Product catalogue. category is Electronics, Home & Kitchen, Fashion, Beauty, Sports, Books or Toys."},
    {"name": "orders", "text": "TABLE orders(order_id INTEGER, customer_id INTEGER, order_date DATE, status VARCHAR, delivery_days DOUBLE). One row per order. status is delivered, cancelled or returned. delivery_days is NULL for cancelled orders."},
    {"name": "order_items", "text": "TABLE order_items(order_item_id INTEGER, order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price DOUBLE). Line items. Line revenue = quantity * unit_price."},
    {"name": "reviews", "text": "TABLE reviews(order_id INTEGER, rating INTEGER). Optional 1-5 star review per order."},
    {"name": "v_orders", "text": (
        "VIEW v_orders (one row per ORDER, all statuses): order_id, customer_id, order_date, month_start (DATE, first day of month), "
        "year (INT), quarter ('Q1'..'Q4'), month (1-12), status, delivery_days, region, customer_type ('new'|'repeat'), "
        "signup_cohort (VARCHAR year), order_revenue (sum of its items), units, n_items, basket_size, rating. "
        "Use for order counts, customers, delivery time, ratings, cancellation rate.")},
    {"name": "v_sales", "text": (
        "VIEW v_sales (one row per ITEM on DELIVERED orders only): order_id, customer_id, order_date, month_start, year, quarter, month, "
        "region, customer_type, signup_cohort, basket_size, delivery_days, rating, category, product_id, quantity, unit_price, "
        "revenue (= quantity * unit_price). Use SUM(revenue) for revenue, COUNT(DISTINCT order_id) for delivered orders, SUM(quantity) for units.")},
]

CORE_SCHEMA = "\n".join(d["text"] for d in SCHEMA_DOCS if d["name"].startswith("v_"))


def chunks() -> list[dict]:
    out = [{"id": f"schema:{d['name']}", "kind": "schema", "text": d["text"]} for d in SCHEMA_DOCS]
    out += [{"id": f"glossary:{k}", "kind": "glossary", "text": f"BUSINESS TERM '{k}': {v}"} for k, v in GLOSSARY.items()]
    return out
