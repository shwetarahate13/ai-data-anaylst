"""DuckDB access layer with a read-only, validated SQL gateway.

Every query an LLM writes goes through `Database.query`, which:
  * accepts a single SELECT/WITH statement only,
  * blocks DDL/DML and file-reading functions,
  * runs on a connection with external file access disabled and configuration locked.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

VIEWS_SQL = """
CREATE OR REPLACE VIEW v_orders AS
WITH base AS (
  SELECT o.order_id, o.customer_id, o.order_date, o.status, o.delivery_days, c.region, c.signup_date,
         ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date, o.order_id) AS order_seq
  FROM orders o JOIN customers c USING (customer_id)
), agg AS (
  SELECT order_id, SUM(quantity * unit_price) AS order_revenue, SUM(quantity) AS units, COUNT(*) AS n_items
  FROM order_items GROUP BY order_id
)
SELECT b.order_id, b.customer_id, b.order_date,
       CAST(date_trunc('month', b.order_date) AS DATE) AS month_start,
       CAST(year(b.order_date) AS INTEGER) AS year,
       'Q' || CAST(quarter(b.order_date) AS VARCHAR) AS quarter,
       CAST(month(b.order_date) AS INTEGER) AS month,
       b.status, b.delivery_days, b.region,
       CASE WHEN b.order_seq = 1 THEN 'new' ELSE 'repeat' END AS customer_type,
       CAST(year(b.signup_date) AS VARCHAR) AS signup_cohort,
       a.order_revenue, a.units, a.n_items,
       CASE WHEN a.n_items = 1 THEN '1 item' WHEN a.n_items = 2 THEN '2 items' ELSE '3+ items' END AS basket_size,
       r.rating
FROM base b JOIN agg a USING (order_id) LEFT JOIN reviews r USING (order_id);

CREATE OR REPLACE VIEW v_sales AS
SELECT o.order_id, o.customer_id, o.order_date, o.month_start, o.year, o.quarter, o.month,
       o.region, o.customer_type, o.signup_cohort, o.basket_size, o.delivery_days, o.rating,
       p.category, oi.product_id, oi.quantity, oi.unit_price, oi.quantity * oi.unit_price AS revenue
FROM v_orders o
JOIN order_items oi USING (order_id)
JOIN products p USING (product_id)
WHERE o.status = 'delivered';
"""

TABLES = ["customers", "products", "orders", "order_items", "reviews"]

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|pragma|call|truncate|grant|vacuum|"
    r"read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text|read_blob|glob|parquet_scan|csv_scan|sniff_csv|"
    r"duckdb_\w+|pragma_\w+|getenv)\b",
    re.I,
)


class UnsafeSQL(ValueError):
    pass


def build_database(data_dir: Path, db_path: Path) -> Path:
    """Load the CSVs into a DuckDB file and create the analytic views."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for t in TABLES:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{Path(data_dir) / (t + '.csv')}', header=true)")
    con.execute("ALTER TABLE orders ALTER order_date TYPE DATE")
    con.execute("ALTER TABLE customers ALTER signup_date TYPE DATE")
    con.execute(VIEWS_SQL)
    con.close()
    return db_path


def validate_sql(sql: str) -> str:
    s = (sql or "").strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    if not s:
        raise UnsafeSQL("empty SQL")
    if ";" in s:
        raise UnsafeSQL("only a single statement is allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise UnsafeSQL("only SELECT / WITH queries are allowed")
    bad = _FORBIDDEN.search(s)
    if bad:
        raise UnsafeSQL(f"keyword or function not allowed: {bad.group(0)}")
    return s


class Database:
    def __init__(self, con: duckdb.DuckDBPyConnection, row_limit: int = 5000):
        self.con = con
        self.row_limit = row_limit
        # Lock the sandbox: no reading arbitrary files, no changing settings afterwards.
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")
        self._values: dict[str, list[str]] | None = None

    @classmethod
    def open(cls, db_path: Path, row_limit: int = 5000) -> "Database":
        return cls(duckdb.connect(str(db_path), read_only=True), row_limit)

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame], row_limit: int = 5000) -> "Database":
        con = duckdb.connect(":memory:")
        for name, df in frames.items():
            con.register(f"_{name}", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
            con.unregister(f"_{name}")
        con.execute("ALTER TABLE orders ALTER order_date TYPE DATE")
        con.execute("ALTER TABLE customers ALTER signup_date TYPE DATE")
        con.execute(VIEWS_SQL)
        return cls(con, row_limit)

    # -- querying -------------------------------------------------------
    def query(self, sql: str, limit: int | None = -1) -> pd.DataFrame:
        """Validated query. limit=-1 uses the default row cap, None disables it (internal pulls)."""
        s = validate_sql(sql)
        cur = self.con.execute(s)
        cols = [d[0] for d in cur.description]
        cap = self.row_limit if limit == -1 else limit
        rows = cur.fetchall() if cap is None else cur.fetchmany(cap + 1)
        df = pd.DataFrame(rows, columns=cols)
        if cap is not None and len(df) > cap:
            df = df.iloc[:cap]
            df.attrs["truncated"] = True
        return df

    def pull(self, sql: str) -> pd.DataFrame:
        """Internal bulk pull used by deterministic tools (no row cap)."""
        return self.query(sql, limit=None)

    def distinct_values(self) -> dict[str, list[str]]:
        if self._values is None:
            q = lambda s: [r[0] for r in self.con.execute(s).fetchall()]
            self._values = {
                "region": q("SELECT DISTINCT region FROM customers ORDER BY 1"),
                "category": q("SELECT DISTINCT category FROM products ORDER BY 1"),
                "status": q("SELECT DISTINCT status FROM orders ORDER BY 1"),
                "customer_type": ["new", "repeat"],
            }
        return self._values

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        lo, hi = self.con.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
        return pd.Timestamp(lo), pd.Timestamp(hi)

    def list_tables(self) -> list[dict]:
        """List all tables and views with types and row counts."""
        rows = self.con.execute(
            "SELECT table_name, table_type FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_type, table_name"
        ).fetchall()
        result = []
        for name, ttype in rows:
            try:
                cnt = self.con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            except Exception:
                cnt = None
            result.append({"table_name": name, "table_type": ttype, "row_count": cnt})
        return result

    def table_schema(self, table_name: str) -> list[dict]:
        """Get column details (name, type, nullable) for a table or view."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        rows = self.con.execute(
            f"SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
        ).fetchall()
        return [{"column_name": r[0], "data_type": r[1], "is_nullable": r[2]} for r in rows]

    def sample_rows(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """Fetch sample rows from a table or view."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        return self.query(f"SELECT * FROM {table_name} LIMIT {limit}")

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """Register an in-memory DataFrame as a queryable table."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid table name: {name}")
        self.con.register(name, df)
