"""Evaluation harness.

  python -m eval.run_eval            # uses the LLM if ANTHROPIC_API_KEY is set, else the rule-based baseline
  python -m eval.run_eval --offline  # force the rule-based baseline

Part 1: execution accuracy on 50 text-to-SQL questions (result-set match against gold SQL).
Part 2: root-cause accuracy on investigations scored against the injected ground truth
        (including negative controls where the correct behaviour is NOT to blame a segment).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst.agent import DataAnalystAgent, InvestigationError  # noqa: E402
from analyst.bootstrap import ensure_database  # noqa: E402
from analyst.config import get_settings  # noqa: E402
from eval.questions import QUESTIONS, RCA_CASES  # noqa: E402


def _norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "null"
    if isinstance(v, (int, float)):
        return f"{round(float(v), 2):.2f}"
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
        return str(v)[:10]
    return str(v).strip().lower()


def results_match(agent_df: pd.DataFrame | None, gold_df: pd.DataFrame) -> bool:
    """Execution-accuracy style check: every gold column must appear (same values) among the agent's columns.

    Order-insensitive, numeric tolerance 0.01, extra helpful columns allowed, column names ignored.
    """
    if agent_df is None or len(agent_df) != len(gold_df):
        return False
    agent_cols = [sorted(_norm(v) for v in agent_df[c]) for c in agent_df.columns]
    for gc in gold_df.columns:
        g = sorted(_norm(v) for v in gold_df[gc])
        if g not in agent_cols:
            return False
    return True


def run_sql_eval(agent: DataAnalystAgent) -> dict:
    rows, by_diff = [], defaultdict(lambda: [0, 0])
    for q in QUESTIONS:
        gold = agent.db.pull(q["gold_sql"])
        t = time.time()
        try:
            res = agent.ask(q["question"])
            ok = results_match(res.df, gold)
            sql, err = res.sql, None
        except Exception as e:  # a crash is a failure, not a reason to stop the eval
            ok, sql, err = False, None, str(e)
        by_diff[q["difficulty"]][0] += int(ok)
        by_diff[q["difficulty"]][1] += 1
        rows.append({"id": q["id"], "difficulty": q["difficulty"], "question": q["question"], "correct": ok,
                     "agent_sql": sql, "gold_sql": q["gold_sql"], "error": err, "seconds": round(time.time() - t, 2)})
    total = sum(r["correct"] for r in rows)
    return {"correct": total, "total": len(rows), "accuracy": total / len(rows),
            "by_difficulty": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in by_diff.items()}, "rows": rows}


def run_rca_eval(agent: DataAnalystAgent) -> dict:
    rows = []
    for case in RCA_CASES:
        exp, notes, ok = case["expect"], [], True
        try:
            inv = agent.investigate(case["question"])
            f = inv.facts
            if "window" in exp and f["window_label"] != exp["window"]:
                ok, _ = False, notes.append(f"window {f['window_label']} != {exp['window']}")
            if "filters" in exp:
                got = (f.get("top_driver") or {}).get("filters", {})
                missing = {k: v for k, v in exp["filters"].items() if got.get(k) != v}
                if missing:
                    ok, _ = False, notes.append(f"driver {got} missing {missing}")
            if "mechanism" in exp and (f.get("mechanism") or {}).get("metric") != exp["mechanism"]:
                ok, _ = False, notes.append(f"mechanism {(f.get('mechanism') or {}).get('metric')} != {exp['mechanism']}")
            if exp.get("no_change_in_direction") and f["matches_direction"]:
                ok, _ = False, notes.append("agent invented a change that did not happen")
            if exp.get("broad_based") and f.get("top_driver"):
                ok, _ = False, notes.append(f"agent blamed a segment for a broad-based move: {f['top_driver']['label']}")
            if any(c.status == "fail" for c in inv.checks):
                ok, _ = False, notes.append("critic reported a failed check")
        except InvestigationError as e:
            ok, _ = False, notes.append(f"error: {e}")
        rows.append({"question": case["question"], "correct": ok, "notes": notes})
    total = sum(r["correct"] for r in rows)
    return {"correct": total, "total": len(rows), "accuracy": total / len(rows), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    settings = get_settings()
    ensure_database(settings)
    agent = DataAnalystAgent.open(settings, use_llm=not args.offline)
    mode = "llm" if agent.llm else "rules"
    print(f"[eval] mode: {agent.mode}")

    sql = run_sql_eval(agent)
    print(f"\nText-to-SQL execution accuracy: {sql['correct']}/{sql['total']} = {sql['accuracy']:.0%}")
    for d in ("easy", "medium", "hard"):
        v = sql["by_difficulty"].get(d)
        if v:
            print(f"  {d:<7} {v['correct']}/{v['total']} = {v['accuracy']:.0%}")
    for r in sql["rows"]:
        if not r["correct"]:
            print(f"  x {r['id']} [{r['difficulty']}] {r['question']}")

    rca = run_rca_eval(agent)
    print(f"\nRoot-cause accuracy (vs injected ground truth, incl. negative controls): {rca['correct']}/{rca['total']} = {rca['accuracy']:.0%}")
    for r in rca["rows"]:
        print(f"  {'ok' if r['correct'] else 'x '} {r['question']}" + (f"  <- {'; '.join(r['notes'])}" if r["notes"] else ""))

    out = {"mode": mode, "model": settings.model if agent.llm else None, "sql": sql, "rca": rca}
    path = Path(__file__).parent / f"results_{mode}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
