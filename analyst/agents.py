"""The agent roles: Planner, Coder (text-to-SQL), Critic helpers and Narrator.

Division of labour (important design choice):
  * LLM  -> *language* work: interpret the question, write SQL for ad-hoc questions, write the summary.
  * Code -> *numbers*: SQL + pandas compute every figure; the Critic re-derives them independently.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from .db import Database, UnsafeSQL
from .knowledge import CORE_SCHEMA
from .llm import LLM, extract_json
from .rules_sql import parse_to_sql, MONTHS
from .sandbox import SandboxError, run_python
from .schema_rag import SchemaRAG
from .utils import Trace, money, pct, quarter_label, sql_date


# ----------------------------------------------------------------------------- Planner
@dataclass
class Plan:
    metric: str = "revenue"                  # revenue | orders
    direction: str = "down"                  # down | up
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None  # [start, end) month-aligned; None -> detect
    baseline_kind: str = "prev"              # prev | yoy
    hypotheses: list[str] = field(default_factory=list)
    source: str = "rules"


DEFAULT_HYPOTHESES = [
    "A specific region, product category or customer group drove the change",
    "A change in order volume (not basket value) explains most of it",
    "An operational metric (delivery time, cancellations, ratings) moved in the affected segment",
]


class Planner:
    def __init__(self, llm: LLM | None):
        self.llm = llm

    def plan(self, question: str, data_range: tuple[pd.Timestamp, pd.Timestamp], trace: Trace) -> Plan:
        plan = self._rules(question, data_range)
        trace.log("Planner", "parsed question (rules)",
                  f"metric={plan.metric}, direction={plan.direction}, window={self._fmt(plan.window)}, baseline={plan.baseline_kind}")
        if self.llm is not None:
            try:
                refined = self._llm_refine(question, data_range, plan)
                trace.log("Planner", "refined plan with LLM", f"hypotheses: {refined.hypotheses}")
                return refined
            except Exception as e:  # never let planning failures kill the run
                trace.log("Planner", "LLM planning failed, keeping rule-based plan", str(e))
        return plan

    @staticmethod
    def _fmt(w):
        return quarter_label(*w) if w else "auto-detect"

    def _rules(self, question: str, data_range) -> Plan:
        q = question.lower()
        p = Plan(hypotheses=list(DEFAULT_HYPOTHESES))
        p.metric = "orders" if re.search(r"\borders?\b|order volume", q) and not re.search(r"revenue|sales", q) else "revenue"
        if re.search(r"increase|grow|grew|rise|rose|jump|surge|spike|up\b|improv", q):
            p.direction = "up"
        if re.search(r"drop|decline|fell|fall|down|decrease|dip|lower|slump|shrink|worse|dropp", q):
            p.direction = "down"
        if re.search(r"year[- ]over[- ]year|\byoy\b|last year|vs\.? same|compared to .*year", q):
            p.baseline_kind = "yoy"
        lo, hi = data_range
        year_m = re.search(r"\b(20\d\d)\b", q)
        year = int(year_m.group(1)) if year_m else None
        qm = re.search(r"\bq([1-4])\b", q) or re.search(r"\b(first|second|third|fourth)\s+quarter", q)
        mm = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", q)
        if qm:
            qn = int(qm.group(1)) if qm.group(1).isdigit() else {"first": 1, "second": 2, "third": 3, "fourth": 4}[qm.group(1)]
            y = year or self._latest_year_with(qn * 3 - 2, 3, lo, hi)
            start = pd.Timestamp(year=y, month=qn * 3 - 2, day=1)
            p.window = (start, start + pd.DateOffset(months=3))
        elif mm:
            m = MONTHS[mm.group(1)]
            y = year or self._latest_year_with(m, 1, lo, hi)
            start = pd.Timestamp(year=y, month=m, day=1)
            p.window = (start, start + pd.DateOffset(months=1))
        elif re.search(r"last quarter|latest quarter|most recent quarter", q):
            end = (hi + pd.Timedelta(days=1)).to_period("Q").start_time
            p.window = (end - pd.DateOffset(months=3), end)
        return p

    @staticmethod
    def _latest_year_with(month: int, n_months: int, lo: pd.Timestamp, hi: pd.Timestamp) -> int:
        for y in range(hi.year, lo.year - 1, -1):
            s = pd.Timestamp(year=y, month=month, day=1)
            if s >= lo.to_period("M").start_time and s + pd.DateOffset(months=n_months) <= hi + pd.Timedelta(days=1):
                return y
        return hi.year

    def _llm_refine(self, question: str, data_range, plan: Plan) -> Plan:
        system = ("You plan business-metric investigations. Reply with ONLY a JSON object: "
                  '{"metric": "revenue"|"orders", "direction": "down"|"up", "period": {"year": int|null, "quarter": 1-4|null, "month": 1-12|null}, '
                  '"baseline": "prev"|"yoy", "hypotheses": [up to 4 short strings]}. '
                  "Hypotheses must be testable with segment breakdowns (region, category, customer type, signup cohort, basket size) "
                  "or operational metrics (delivery days, cancellation rate, rating).")
        user = f"Data covers {data_range[0].date()} to {data_range[1].date()}.\nQuestion: {question}"
        obj = extract_json(self.llm.complete(system, user, max_tokens=500))
        out = Plan(**{**plan.__dict__})
        if obj.get("metric") in ("revenue", "orders"):
            out.metric = obj["metric"]
        if obj.get("direction") in ("down", "up"):
            out.direction = obj["direction"]
        if obj.get("baseline") in ("prev", "yoy"):
            out.baseline_kind = obj["baseline"]
        per = obj.get("period") or {}
        lo, hi = data_range
        try:
            if per.get("quarter"):
                qn = int(per["quarter"])
                y = int(per.get("year") or self._latest_year_with(qn * 3 - 2, 3, lo, hi))
                s = pd.Timestamp(year=y, month=qn * 3 - 2, day=1)
                out.window = (s, s + pd.DateOffset(months=3))
            elif per.get("month"):
                m = int(per["month"])
                y = int(per.get("year") or self._latest_year_with(m, 1, lo, hi))
                s = pd.Timestamp(year=y, month=m, day=1)
                out.window = (s, s + pd.DateOffset(months=1))
        except (ValueError, TypeError):
            pass
        hyps = [str(h)[:160] for h in (obj.get("hypotheses") or [])][:4]
        out.hypotheses = hyps or out.hypotheses
        out.source = "llm"
        return out


# ----------------------------------------------------------------------------- Coder
@dataclass
class AskResult:
    question: str
    sql: str | None
    df: pd.DataFrame | None
    answer: str
    mode: str                                # llm | rules | none
    attempts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: Trace = field(default_factory=Trace)

    def to_dict(self) -> dict:
        return {"question": self.question, "sql": self.sql, "answer": self.answer, "mode": self.mode,
                "columns": list(self.df.columns) if self.df is not None else [],
                "rows": self.df.head(200).astype(object).where(self.df.head(200).notna(), None).values.tolist() if self.df is not None else [],
                "attempts": self.attempts, "warnings": self.warnings, "trace": self.trace.to_list()}


SQL_SYSTEM = f"""You are a senior analytics engineer writing DuckDB SQL for a business analyst.
Rules:
- Reply with ONLY a JSON object: {{"sql": "<one SELECT or WITH statement>", "python": null}}.
  ("python" is optional pandas post-processing on a DataFrame named df; set result = ...; use only when SQL cannot express it.)
- Prefer the views v_orders (order grain) and v_sales (delivered item grain). Never modify data.
- Follow the business definitions provided. Revenue = delivered only. 'orders' = all statuses.
- Use exact category/region spellings from the value list. Alias columns clearly. ROUND money to 2 decimals.
- Dates: use DATE 'YYYY-MM-DD'. Fiscal = calendar year. Quarter columns hold 'Q1'..'Q4'.

Core schema:
{CORE_SCHEMA}
"""


class Coder:
    """Text-to-SQL agent: schema RAG -> draft -> execute -> self-repair on errors / empty results."""

    def __init__(self, db: Database, llm: LLM | None, rag: SchemaRAG, max_retries: int = 3):
        self.db, self.llm, self.rag, self.max_retries = db, llm, rag, max_retries

    # -- deterministic SQL templates used by the investigation pipeline ----
    @staticmethod
    def sql_monthly(metric: str) -> str:
        if metric == "orders":
            return "SELECT month_start, COUNT(*) AS value FROM v_orders GROUP BY 1 ORDER BY 1"
        return "SELECT month_start, SUM(revenue) AS value FROM v_sales GROUP BY 1 ORDER BY 1"

    @staticmethod
    def sql_items() -> str:
        return ("SELECT order_id, order_date, region, category, customer_type, signup_cohort, basket_size, revenue "
                "FROM v_sales")

    @staticmethod
    def sql_orders() -> str:
        return ("SELECT order_id, order_date, region, customer_type, signup_cohort, basket_size, status, "
                "delivery_days, rating, order_revenue FROM v_orders")

    # -- ad-hoc questions -------------------------------------------------
    def answer(self, question: str, trace: Trace | None = None) -> AskResult:
        trace = trace or Trace()
        if self.llm is None:
            return self._answer_rules(question, trace)
        return self._answer_llm(question, trace)

    def _answer_rules(self, question: str, trace: Trace) -> AskResult:
        v = self.db.distinct_values()
        sql, note = parse_to_sql(question, v["region"], v["category"])
        if sql is None:
            trace.log("Coder", "rule-based parser could not handle the question", note)
            return AskResult(question, None, None,
                             "I can't answer that offline. Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) to enable the LLM text-to-SQL agent.",
                             "none", trace=trace)
        trace.log("Coder", "generated SQL (rules)", note, sql)
        df = self.db.query(sql)
        return self._finish(question, sql, df, "rules", [{"sql": sql, "error": None}], trace)

    def _answer_llm(self, question: str, trace: Trace) -> AskResult:
        v = self.db.distinct_values()
        ctx = self.rag.context(question, k=6)
        base_user = (f"Relevant schema and business definitions:\n{ctx}\n\n"
                     f"Allowed values: {json.dumps(v)}\n\nQuestion: {question}")
        attempts: list[dict] = []
        feedback = ""
        df, sql = None, None
        for i in range(self.max_retries):
            try:
                raw = self.llm.complete(SQL_SYSTEM, base_user + feedback, max_tokens=900)
                obj = extract_json(raw)
                sql, py = obj.get("sql"), obj.get("python")
                trace.log("Coder", f"draft SQL (attempt {i + 1})", "", sql)
                df = self.db.query(sql)
                if py:
                    try:
                        out = run_python(py, df)
                        df = out if isinstance(out, pd.DataFrame) else pd.DataFrame({"result": [out]})
                        trace.log("Coder", "post-processed in sandbox", py)
                    except SandboxError as e:
                        trace.log("Coder", "sandbox post-processing failed (ignored)", str(e))
            except (UnsafeSQL, ValueError, KeyError, TypeError) as e:
                attempts.append({"sql": sql, "error": str(e)})
                trace.log("Coder", "attempt failed", str(e))
                feedback = f"\n\nYour previous reply failed.\nSQL: {sql}\nError: {e}\nReturn a corrected JSON object."
                df = None
                continue
            except Exception as e:  # DuckDB binder/parser errors etc.
                attempts.append({"sql": sql, "error": str(e)})
                trace.log("Coder", "SQL error -> self-repair", str(e))
                feedback = f"\n\nYour previous SQL failed.\nSQL: {sql}\nError: {e}\nFix it and return the corrected JSON object."
                df = None
                continue
            if df.empty and i < self.max_retries - 1:
                attempts.append({"sql": sql, "error": "zero rows"})
                trace.log("Critic", "empty result -> asking Coder to re-check filters")
                feedback = f"\n\nYour SQL returned zero rows:\n{sql}\nCheck filter values/spelling against the allowed values and try again."
                continue
            attempts.append({"sql": sql, "error": None})
            break
        if df is None:
            return AskResult(question, sql, None, "I could not produce a valid query for that question.", "llm",
                             attempts, ["all attempts failed"], trace)
        return self._finish(question, sql, df, "llm", attempts, trace)

    def _finish(self, question, sql, df, mode, attempts, trace) -> AskResult:
        warnings = []
        if df.empty:
            warnings.append("query returned no rows")
        if df.attrs.get("truncated"):
            warnings.append(f"result truncated to {len(df)} rows")
        if len(df) == 1 and len(df.columns) == 1:
            val = df.iloc[0, 0]
            answer = f"{df.columns[0]} = {val:,.2f}" if isinstance(val, float) else f"{df.columns[0]} = {val}"
        elif df.empty:
            answer = "No rows matched."
        else:
            answer = f"Returned {len(df)} row(s)."
        trace.log("Critic", "checked result", "; ".join(warnings) or "ok")
        return AskResult(question, sql, df, answer, mode, attempts, warnings, trace)


# ----------------------------------------------------------------------------- Narrator
_NUM = re.compile(r"(?<![\w.])[-−+]?\$?\d[\d,]*\.?\d*\s?(?:[KkMm]\b|%)?")


def _candidates(obj) -> set[float]:
    """All numbers in the verified facts, in every unit the narrator may plausibly use."""
    vals: set[float] = set()

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            f = abs(float(x))
            vals.update({f, f * 100, f / 1e3, f / 1e6})
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for m in re.findall(r"\d+\.?\d*", x):
                vals.add(float(m))
    walk(obj)
    return vals


def ungrounded_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in `text` that cannot be traced to `facts` (within rounding). Empty list = fully grounded."""
    cands = _candidates(facts)
    bad = []
    for m in _NUM.finditer(text):
        tok = m.group(0).strip().rstrip(".,")
        digits = re.sub(r"[^\d.]", "", tok.replace(",", ""))
        if not digits or digits == ".":
            continue
        try:
            v = float(digits)
        except ValueError:
            continue
        # skip calendar years, and quarter numbers like Q3
        if 1900 <= v <= 2100 and float(v).is_integer() and "." not in tok and "$" not in tok:
            continue
        before = text[max(0, m.start() - 1):m.start()]
        if before in ("Q", "q") or (m.start() > 0 and text[m.start() - 1] in "Qq"):
            continue
        if float(v).is_integer() and v <= 10 and not re.search(r"[%$KkMm]", tok):
            continue  # small counts like "3 steps"
        unit = 1.0
        if tok.lower().rstrip().endswith("k"):
            unit = 1e3
        elif tok.lower().rstrip().endswith("m"):
            unit = 1e6
        target = v * unit
        ok = any(abs(c - target) <= max(0.0051 * max(abs(c), 1.0), 0.06) or abs(c - v) <= max(0.0051 * max(abs(c), 1.0), 0.06)
                 for c in cands)
        if not ok:
            bad.append(tok)
    return bad


class Narrator:
    def __init__(self, llm: LLM | None, currency: str = "$"):
        self.llm, self.cur = llm, currency

    def write(self, facts: dict, trace: Trace) -> str:
        template = self.template(facts)
        if self.llm is None:
            trace.log("Narrator", "wrote summary (template)")
            return template
        try:
            system = ("You are a senior analyst writing an executive summary for a business reader. Use ONLY the facts in the JSON; "
                      "do not invent numbers, causes or recommendations beyond them. Use markdown, under 180 words: a bold headline sentence, "
                      "then 3-4 short bullets (where it came from, likely mechanism, context, confidence/caveats). "
                      "State clearly that the mechanism is an association, not proof of causation. Mention the what-if only if present, labelled as an estimate.")
            text = self.llm.complete(system, json.dumps(facts, default=str), max_tokens=700).strip()
            bad = ungrounded_numbers(text, facts)
            if bad:
                trace.log("Narrator", "LLM summary rejected: ungrounded numbers -> using template", ", ".join(bad[:6]))
                return template
            trace.log("Narrator", "wrote summary (LLM, numbers grounded)")
            return text
        except Exception as e:
            trace.log("Narrator", "LLM narration failed -> template", str(e))
            return template

    # -- deterministic narrative -------------------------------------------
    def template(self, f: dict) -> str:
        c = self.cur
        m = f["metric"]
        fmt = (lambda x: money(x, c)) if m == "revenue" else (lambda x: f"{x:,.0f}")
        sfmt = lambda x: ("+" if x > 0 else "") + fmt(x)
        word = "Revenue" if m == "revenue" else "Orders"
        w, b = f["window_label"], f["baseline_label"]
        up = f["delta"] >= 0
        lines = []
        if not f["matches_direction"]:
            asked = "fall" if f["direction"] == "down" else "rise"
            lines.append(f"**{word} did not {asked} in {w}: it moved {pct(f['pct_change'], True)} vs {b}** "
                         f"({fmt(f['baseline_total'])} to {fmt(f['window_total'])}), so there is no such change to explain.")
            if f.get("top_movers"):
                mv = f["top_movers"][0]
                lines.append(f"- The weakest segment was {mv['label']} ({fmt(mv['delta'])}), but it was outweighed by growth elsewhere.")
        else:
            verb = "rose" if up else "fell"
            lines.append(f"**{word} {verb} {pct(abs(f['pct_change']))} in {w} vs {b}** "
                         f"({fmt(f['baseline_total'])} to {fmt(f['window_total'])}, {fmt(f['delta'])}).")
            td = f.get("top_driver")
            if td:
                share = abs(td["share_net"])
                rest = f["rest_delta"]
                if share > 1.0:
                    lines.append(f"- **Where it came from:** {td['label']} moved {sfmt(td['delta'])}, more than the entire net change "
                                 f"({sfmt(f['delta'])}), because the rest of the business moved {sfmt(rest)}.")
                else:
                    lines.append(f"- **Where it came from:** {td['label']} account for {sfmt(td['delta'])}, i.e. {pct(share, digits=0)} of the net change; "
                                 f"the rest of the business moved {sfmt(rest)}.")
            else:
                mv = f.get("top_movers") or []
                names = ", ".join(f"{x['label']} ({fmt(x['delta'])})" for x in mv[:2])
                lines.append("- **Where it came from:** the change is broad-based. No single region, category, customer group or cohort "
                             "carries more than its proportional share" + (f"; the largest movers were {names}." if names else "."))
            vp = f.get("volume_price")
            if vp and m == "revenue":
                vol_word = "Lower order volume" if vp["volume_effect"] < 0 else "Higher order volume"
                lines.append(f"- **Volume vs value:** {vol_word.lower()} accounts for {sfmt(vp['volume_effect'])}; "
                             f"basket value (AOV {c}{vp['aov_base']:,.0f} to {c}{vp['aov_window']:,.0f}) accounts for {sfmt(vp['price_effect'])}.")
            mech = f.get("mechanism")
            if mech:
                rt = f.get("rating")
                rating_txt = (f"Average review rating in that segment also moved from {rt['slice_base']:.1f} to {rt['slice_window']:.1f} stars. "
                              if rt else "")
                unit = {"delivery_days": " days", "rating": " stars", "cancelled": ""}[mech["metric"]]
                nice = {"delivery_days": "average delivery time", "rating": "average review rating", "cancelled": "cancellation rate"}[mech["metric"]]
                sv = lambda x: f"{(0.0 if abs(x) < 0.05 else x):.1f}{unit}" if mech["metric"] != "cancelled" else pct(x)
                lines.append(f"- **Likely mechanism:** {nice} for that segment went from {sv(mech['slice_base'])} to {sv(mech['slice_window'])} "
                             f"({'+' if mech['slice_change'] >= 0 else ''}{sv(mech['slice_change'])}), "
                             f"versus {'+' if mech['rest_change'] >= -0.05 else ''}{sv(mech['rest_change'])} "
                             f"for the rest of the business (z = {mech['z']:.0f}). "
                             f"{rating_txt}This is a strong association, not proof of causation.")
            yoy = f.get("yoy")
            if yoy:
                tail = ("so part of the move is seasonal." if (not up and yoy["window"] > 0.05) else
                        "so this is not just seasonality." if not up else ".")
                prior = (f" (the preceding period was {pct(yoy['prior'], True)})" if yoy.get("prior") is not None else "")
                lines.append(f"- **Context:** versus the same period last year, {w} {word.lower()} was {pct(yoy['window'], True)}{prior}, {tail}")
            wi = f.get("whatif")
            if wi:
                lines.append(f"- **What-if (estimate):** cutting delivery time by {pct(wi['reduction'], digits=0)} in {wi['region']} could have added about "
                             f"{money(wi['extra_revenue'], c)} (range {money(wi['revenue_low'], c)} to {money(wi['revenue_high'], c)}) of repeat-customer revenue. "
                             f"{'This extrapolates beyond observed delivery times.' if wi['extrapolation'] else ''}".rstrip())
        conf = f["confidence"]
        lines.append(f"- **Confidence: {conf}.** {f['checks_passed']} of {f['checks_total']} verification checks passed "
                     f"(totals re-derived from raw tables with independent SQL).")
        return "\n".join(lines)
