import pandas as pd

from analyst.agent import DataAnalystAgent

from .conftest import FakeLLM


def test_rules_mode_matches_gold(agent, db):
    res = agent.ask("What was the total revenue in 2024?")
    gold = db.pull("SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE year = 2024").iloc[0, 0]
    assert res.mode == "rules" and abs(res.df.iloc[0, 0] - gold) < 0.01


def test_rules_mode_admits_what_it_cannot_do(agent):
    res = agent.ask("Predict next year's churn using a neural net")
    assert res.sql is None and "ANTHROPIC_API_KEY" in res.answer


def test_llm_self_repairs_bad_sql(db):
    llm = FakeLLM([
        '{"sql": "SELECT SUM(revnue) FROM v_sales"}',                      # typo -> binder error
        '{"sql": "SELECT ROUND(SUM(revenue),2) AS revenue FROM v_sales"}',  # repaired
    ])
    res = DataAnalystAgent(db, llm).ask("total revenue?")
    assert res.mode == "llm" and len(res.attempts) == 2 and res.attempts[0]["error"] and res.attempts[1]["error"] is None
    assert "failed" in llm.calls[1][1]  # the error was fed back to the model
    assert res.df.iloc[0, 0] > 1_000_000


def test_llm_empty_rows_trigger_retry(db):
    llm = FakeLLM(['{"sql": "SELECT region FROM v_orders WHERE region = \'Westt\' LIMIT 3"}',
                   '{"sql": "SELECT DISTINCT region FROM v_orders WHERE region = \'West\'"}'])
    res = DataAnalystAgent(db, llm).ask("which region is west")
    assert len(res.attempts) == 2 and res.attempts[0]["error"] == "zero rows" and res.df.iloc[0, 0] == "West"


def test_llm_cannot_write_destructive_sql(db):
    llm = FakeLLM(['{"sql": "DROP TABLE orders"}', '{"sql": "DELETE FROM orders"}', '{"sql": "SELECT COUNT(*) FROM orders"}'])
    res = DataAnalystAgent(db, llm).ask("clean up the database")
    assert [a["error"] is None for a in res.attempts] == [False, False, True]
    assert db.query("SELECT COUNT(*) FROM orders").iloc[0, 0] > 60000  # still intact


def test_llm_sandbox_postprocessing(db):
    llm = FakeLLM(['{"sql": "SELECT region, SUM(revenue) AS rev FROM v_sales GROUP BY region", "python": "result = df.assign(share=df.rev / df.rev.sum()).sort_values(\'share\', ascending=False)"}'])
    res = DataAnalystAgent(db, llm).ask("revenue share by region")
    assert abs(res.df["share"].sum() - 1) < 1e-9 and res.df["share"].is_monotonic_decreasing


def test_llm_all_attempts_fail_gracefully(db):
    res = DataAnalystAgent(db, FakeLLM(["not json"] * 3)).ask("???")
    assert res.df is None and "could not" in res.answer
