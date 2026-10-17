import json

import pandas as pd
import pytest

from analyst.db import UnsafeSQL, validate_sql
from analyst.llm import extract_json
from analyst.sandbox import SandboxError, run_python
from analyst.schema_rag import SchemaRAG
from analyst.tools.anomaly import detect_anomalies


# ---------------------------------------------------------------- SQL gateway
@pytest.mark.parametrize("sql", ["DROP TABLE orders", "SELECT 1; SELECT 2", "DELETE FROM orders", "UPDATE orders SET status='x'",
                                 "SELECT * FROM read_csv('/etc/passwd')", "COPY orders TO '/tmp/x.csv'", "ATTACH 'x.db'", "PRAGMA database_list", ""])
def test_unsafe_sql_rejected(sql):
    with pytest.raises(UnsafeSQL):
        validate_sql(sql)


def test_safe_sql_accepted(db):
    assert validate_sql("SELECT 1;") == "SELECT 1"
    assert validate_sql("WITH a AS (SELECT 1 x) SELECT * FROM a")
    assert db.query("SELECT COUNT(*) AS n FROM orders").iloc[0, 0] > 1000


def test_external_access_locked(db):
    with pytest.raises(Exception):
        db.con.execute("SELECT * FROM read_csv('/etc/passwd')")
    with pytest.raises(Exception):
        db.con.execute("SET enable_external_access = true")


def test_row_cap_flags_truncation(db):
    df = db.query("SELECT * FROM orders", limit=10)
    assert len(df) == 10 and df.attrs.get("truncated")


# ---------------------------------------------------------------- sandbox
def test_sandbox_runs_pandas():
    out = run_python("result = df.assign(y=df.x * 2)", pd.DataFrame({"x": [1, 2]}))
    assert out["y"].tolist() == [2, 4]


@pytest.mark.parametrize("code", ["import os\nresult = 1", "result = open('/etc/passwd').read()", "result = ().__class__", "import subprocess"])
def test_sandbox_blocks_dangerous_code(code):
    with pytest.raises(SandboxError):
        run_python(code, pd.DataFrame({"x": [1]}))


def test_sandbox_timeout():
    with pytest.raises(SandboxError):
        run_python("while True:\n    pass", pd.DataFrame({"x": [1]}), timeout=2)


# ---------------------------------------------------------------- RAG / json
def test_rag_retrieves_relevant_chunks():
    rag = SchemaRAG()
    assert any("cancellation" in d["id"] for d in rag.retrieve("what is the cancellation rate"))
    assert any(d["id"] == "schema:v_orders" for d in rag.retrieve("average delivery days per order"))
    assert any("churn" in d["id"] for d in rag.retrieve("how many customers churned"))


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"sql": "SELECT 1", "n": {"a": "}"}}\n```') == {"sql": "SELECT 1", "n": {"a": "}"}}
    with pytest.raises(ValueError):
        extract_json("no json here")


# ---------------------------------------------------------------- anomaly detection
def test_detects_injected_q3_window(db):
    s = db.pull("SELECT month_start, SUM(revenue) v FROM v_sales GROUP BY 1 ORDER BY 1").set_index("month_start")["v"]
    res = detect_anomalies(s, "down")
    assert len(res.windows) == 1
    w = res.windows[0]
    assert w.start == pd.Timestamp("2024-07-01") and w.end == pd.Timestamp("2024-10-01")
    assert detect_anomalies(s, "up").windows == []  # holiday peaks are seasonal, not anomalies


def test_ground_truth_file_matches_generator(frames):
    truth = frames[1]
    assert truth["region"] == "West" and truth["mechanism"] == "delivery_days"
