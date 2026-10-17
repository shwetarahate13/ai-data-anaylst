"""Command line interface:  python -m analyst <command>"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .agent import DataAnalystAgent, InvestigationError
from .bootstrap import ensure_database
from .config import get_settings
from .report import save_report


def _agent(args) -> DataAnalystAgent:
    settings = get_settings()
    ensure_database(settings)
    agent = DataAnalystAgent.open(settings, use_llm=not args.offline)
    print(f"[mode] {agent.mode}\n", file=sys.stderr)
    return agent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="analyst", description="Autonomous AI Data Analyst Agent")
    ap.add_argument("--offline", action="store_true", help="ignore ANTHROPIC_API_KEY and use the rule-based fallbacks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data", help="(re)generate the synthetic dataset and DuckDB file")
    g.add_argument("--force", action="store_true")

    i = sub.add_parser("investigate", help="root-cause investigation of a metric change")
    i.add_argument("question")
    i.add_argument("--report", action="store_true", help="also write Markdown + HTML reports to ./reports")
    i.add_argument("--trace", action="store_true", help="print the full agent trace")
    i.add_argument("--json", action="store_true")

    a = sub.add_parser("ask", help="ad-hoc question answered with SQL")
    a.add_argument("question")

    w = sub.add_parser("whatif", help="delivery-time what-if simulation")
    w.add_argument("--reduction", type=float, default=0.20, help="fractional cut in delivery time (0.2 = 20%%)")
    w.add_argument("--region", default=None)

    sub.add_parser("eval", help="run the evaluation harness (see eval/run_eval.py)")

    args = ap.parse_args(argv)

    if args.cmd == "generate-data":
        ensure_database(get_settings(), force=True)
        return 0
    if args.cmd == "eval":
        from eval.run_eval import main as run_eval
        return run_eval(["--offline"] if args.offline else [])

    agent = _agent(args)
    if args.cmd == "investigate":
        try:
            inv = agent.investigate(args.question)
        except InvestigationError as e:
            print(f"Cannot investigate: {e}")
            return 2
        if args.json:
            print(json.dumps(inv.to_dict(), indent=2, default=str))
        else:
            print(inv.narrative)
            print("\nVerification:")
            for c in inv.checks:
                print(f"  [{c.status.upper():4}] {c.name}: {c.detail}")
        if args.trace:
            print("\nTrace:")
            for e in inv.trace.to_list():
                print(f"  {e['t']:>5}s {e['agent']:<9} {e['action']}" + (f" - {e['detail'][:140]}" if e["detail"] else ""))
        if args.report:
            paths = save_report(inv, get_settings().reports_dir)
            print(f"\nReport written: {paths['html']}")
        return 0
    if args.cmd == "ask":
        res = agent.ask(args.question)
        if res.sql:
            print(f"SQL ({res.mode}):\n  {res.sql}\n")
        if res.df is not None:
            with pd.option_context("display.width", 120, "display.max_rows", 30):
                print(res.df.head(30).to_string(index=False))
        print(f"\n{res.answer}")
        for wmsg in res.warnings:
            print(f"warning: {wmsg}")
        return 0
    if args.cmd == "whatif":
        try:
            out = agent.whatif(args.reduction, args.region)
        except InvestigationError as e:
            print(f"Cannot simulate: {e}")
            return 2
        c = get_settings().currency
        print(f"What if delivery time fell {args.reduction:.0%} in {out['region']} during {out['window_label']}?")
        print(f"  elasticity: each extra delivery day changes repeat orders by {out['beta'] * 100:+.1f}% (t = {out['beta_t']:.1f})")
        print(f"  delivery time: {out['avg_days_before']:.1f} -> {out['avg_days_after']:.1f} days")
        print(f"  extra repeat orders: ~{out['extra_repeat_orders']:,.0f}; extra revenue: ~{c}{out['extra_revenue']:,.0f} "
              f"(range {c}{out['revenue_low']:,.0f} to {c}{out['revenue_high']:,.0f})")
        if out["extrapolation"]:
            print("  warning: scenario goes below delivery times seen in the data; treat as extrapolation")
        print(f"  note: {out['caveat']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
