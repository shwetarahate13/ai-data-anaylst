"""Orchestrator: the agent loop that investigates a business question end to end.

    Planner -> Coder (SQL pulls) -> Analyst tools (anomaly, attribution, mechanism, what-if)
            -> Critic (independent verification; recompute on failure) -> Narrator

Every step is appended to a Trace so the whole investigation is inspectable in the UI/CLI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .agents import AskResult, Coder, Narrator, Plan, Planner
from .config import Settings, get_settings
from .critic import Check, Critic
from .db import Database
from .llm import LLM, get_llm
from .schema_rag import SchemaRAG
from .tools import whatif as whatif_tool
from .tools.anomaly import detect_anomalies
from .tools.drivers import ITEM_DIMS, ORDER_DIMS, attribute, mechanism_scan, segment_label, volume_price_split
from .utils import Trace, money, quarter_label


class InvestigationError(ValueError):
    """Raised for questions that cannot be investigated (period outside the data, no baseline, ...)."""


@dataclass
class Investigation:
    question: str
    plan: Plan
    facts: dict
    narrative: str
    checks: list[Check]
    confidence: str
    trace: Trace
    data: dict = field(default_factory=dict)  # DataFrames/Series used for charts

    def to_dict(self) -> dict:
        return {
            "question": self.question, "narrative": self.narrative, "confidence": self.confidence,
            "facts": self.facts, "checks": [c.to_dict() for c in self.checks], "trace": self.trace.to_list(),
            "plan": {"metric": self.plan.metric, "direction": self.plan.direction, "baseline": self.plan.baseline_kind,
                     "hypotheses": self.plan.hypotheses, "source": self.plan.source},
        }


def _tag(df: pd.DataFrame, baseline, window) -> pd.DataFrame:
    d = df.copy()
    d["period"] = "other"
    d.loc[(d["order_date"] >= baseline[0]) & (d["order_date"] < baseline[1]), "period"] = "base"
    d.loc[(d["order_date"] >= window[0]) & (d["order_date"] < window[1]), "period"] = "window"
    return d


def _overlap(w, a) -> bool:
    return a.start < w[1] and w[0] < a.end


class DataAnalystAgent:
    def __init__(self, db: Database, llm: LLM | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.db, self.llm = db, llm
        self.rag = SchemaRAG()
        self.coder = Coder(db, llm, self.rag, self.settings.max_sql_retries)
        self.planner = Planner(llm)
        self.critic = Critic(db)
        self.narrator = Narrator(llm, self.settings.currency)

    @classmethod
    def open(cls, settings: Settings | None = None, use_llm: bool = True) -> "DataAnalystAgent":
        settings = settings or get_settings()
        db = Database.open(settings.db_path, settings.row_limit)
        return cls(db, get_llm(settings) if use_llm else None, settings)

    @property
    def mode(self) -> str:
        return f"LLM ({self.settings.model})" if self.llm else "offline (rule-based planner/SQL, template narrator)"

    # ------------------------------------------------------------------ ask
    def ask(self, question: str) -> AskResult:
        trace = Trace()
        trace.log("Planner", "route", "ad-hoc question -> text-to-SQL")
        return self.coder.answer(question, trace)

    # ------------------------------------------------------------- whatif
    def whatif(self, reduction: float = 0.20, region: str | None = None, window=None) -> dict:
        orders = self._pull_orders()
        if window is None:
            monthly = self._monthly("revenue")
            wins = detect_anomalies(monthly, "down").windows
            if not wins:
                raise InvestigationError("No anomalous period found; pass an explicit window.")
            w = min(wins, key=lambda x: x.severity)
            window = (w.start, w.end)
        out = whatif_tool.simulate(orders, window, reduction, region)
        out["window_label"] = quarter_label(*window)
        return out

    # ------------------------------------------------------------ helpers
    def _monthly(self, metric: str) -> pd.Series:
        ms = self.db.pull(self.coder.sql_monthly(metric))
        return pd.Series(ms["value"].astype(float).to_numpy(), index=pd.to_datetime(ms["month_start"]), name="value")

    def _pull_orders(self) -> pd.DataFrame:
        o = self.db.pull(self.coder.sql_orders())
        o["order_date"] = pd.to_datetime(o["order_date"])
        return o

    def _pull_items(self) -> pd.DataFrame:
        i = self.db.pull(self.coder.sql_items())
        i["order_date"] = pd.to_datetime(i["order_date"])
        return i

    @staticmethod
    def _baseline(window, kind):
        ws, we = window
        if kind == "yoy":
            return ws - pd.DateOffset(years=1), we - pd.DateOffset(years=1)
        months = (we.year - ws.year) * 12 + (we.month - ws.month)
        return ws - pd.DateOffset(months=months), ws

    # -------------------------------------------------------- investigate
    def investigate(self, question: str) -> Investigation:
        trace = Trace()
        lo, hi = self.db.date_range()
        plan = self.planner.plan(question, (lo, hi), trace)
        metric = plan.metric

        sql = self.coder.sql_monthly(metric)
        trace.log("Coder", f"pull monthly {metric} series", sql=sql)
        monthly = self._monthly(metric)

        an = detect_anomalies(monthly, plan.direction)
        trace.log("Analyst", f"anomaly scan ({an.method})",
                  "; ".join(f"{quarter_label(w.start, w.end)} (t={w.severity:.1f}, IsolationForest agrees on {w.corroborated:.0%})"
                            for w in an.windows) or "no unusual period found")

        window = plan.window
        if window is None:
            if not an.windows:
                raise InvestigationError(f"No unusual {'drop' if plan.direction == 'down' else 'increase'} in {metric} was found. "
                                         "Name a period (e.g. 'Q3 2024') to investigate a specific one.")
            best = max(an.windows, key=lambda w: abs(w.severity))
            window = (best.start, best.end)
            trace.log("Planner", "no period in question -> using the strongest detected window", quarter_label(*window))
        baseline = self._baseline(window, plan.baseline_kind)
        if window[1] > hi + pd.Timedelta(days=1) or window[0] < lo:
            raise InvestigationError(f"The data only covers {lo.date()} to {hi.date()}; {quarter_label(*window)} is outside it.")
        if baseline[0] < lo.to_period('M').start_time:
            raise InvestigationError(f"There is no baseline data before {quarter_label(*window)} "
                                     f"({quarter_label(*baseline)} is outside the data range).")
        wl, bl = quarter_label(*window), quarter_label(*baseline)
        trace.log("Planner", "comparison set", f"explain {wl} vs {bl} ({'year-over-year' if plan.baseline_kind == 'yoy' else 'previous period'})")
        anomaly_window = next((w for w in an.windows if _overlap(window, w)), None)

        # analysis + verification, with one recompute round if the Critic finds a mismatch
        checks: list[Check] = []
        for attempt in (1, 2):
            state = self._analyze(plan, window, baseline, trace, attempt)
            checks = self.critic.verify(metric=metric, window=window, baseline=baseline, attr=state["attr"], monthly=monthly,
                                        slice_counts=state["slice_counts"], mechanism=state["mechanism"], anomaly_window=anomaly_window)
            fails = [c for c in checks if c.status == "fail"]
            trace.log("Critic", f"verification round {attempt}",
                      f"{sum(c.status == 'pass' for c in checks)} pass, {sum(c.status == 'warn' for c in checks)} warn, {len(fails)} fail")
            if not fails:
                break
            trace.log("Critic", "mismatch found -> discarding cached frames and recomputing from fresh SQL", "; ".join(c.name for c in fails))
        confidence = Critic.confidence(checks)

        facts = self._facts(plan, window, baseline, wl, bl, monthly, state, anomaly_window, checks, confidence)
        narrative = self.narrator.write(facts, trace)
        data = {"monthly": monthly, "window": window, "baseline": baseline, "attr": state["attr"], "mechanism": state["mechanism"],
                "weekly": state["weekly"], "whatif": state["whatif"], "anomaly": an}
        return Investigation(question, plan, facts, narrative, checks, confidence, trace, data)

    def _analyze(self, plan: Plan, window, baseline, trace: Trace, attempt: int) -> dict:
        metric = plan.metric
        orders = self._tag_orders(window, baseline)
        trace.log("Coder", "pull order-level frame" + (" (recompute)" if attempt > 1 else ""), sql=self.coder.sql_orders())
        if metric == "revenue":
            trace.log("Coder", "pull item-level frame", sql=self.coder.sql_items())
            frame = _tag(self._pull_items(), baseline, window)
            value_col, dims = "revenue", ITEM_DIMS
        else:
            frame = orders.copy()
            frame["n"] = 1.0
            value_col, dims = "n", ORDER_DIMS
        mismatch_dir = None
        attr = attribute(frame, value_col, dims, plan.direction)
        total_ok = (attr.delta < 0) if plan.direction == "down" else (attr.delta > 0)
        if not total_ok:
            attr = attribute(frame, value_col, dims, plan.direction, max_depth=0)  # nothing to explain: don't invent a driver
            mismatch_dir = True
        trace.log("Analyst", "driver attribution",
                  f"{metric} {attr.base_total:,.0f} -> {attr.window_total:,.0f} ({attr.pct:+.1%}); "
                  + (f"drill path: {' > '.join(f'{s.dim}={s.segment}' for s in attr.path)}" if attr.path else
                     ("direction does not match the question" if mismatch_dir else "no segment carries more than its proportional share (broad-based)")))
        vp = None
        if metric == "revenue":
            vp = volume_price_split(orders[orders["status"] == "delivered"])
            trace.log("Analyst", "volume vs price/mix split", f"volume {vp['volume_effect']:,.0f}, price/mix {vp['price_effect']:,.0f}")

        mech, weekly, slice_counts, wi = None, None, None, None
        order_filters = {k: v for k, v in attr.filters.items() if k in ORDER_DIMS}
        if attr.filters and order_filters:
            mech = mechanism_scan(orders, attr.filters)
            if not mech.empty:
                trace.log("Analyst", "mechanism scan (difference-in-differences vs rest of business)",
                          "; ".join(f"{r.metric}: slice {r.slice_change:+.2f} vs rest {r.rest_change:+.2f}, z={r.z:.1f}" for r in mech.itertuples()))
            m = pd.Series(True, index=orders.index)
            for k, v in order_filters.items():
                m &= orders[k] == v
            slice_counts = {"base": int((m & (orders["period"] == "base")).sum()), "window": int((m & (orders["period"] == "window")).sum())}
            weekly = self._weekly(orders, m)
        if mech is not None and not mech.empty and attr.filters:
            cause = mech[(mech["metric"].isin(["delivery_days", "cancelled"])) & mech["significant"]]
            if not cause.empty and cause.iloc[0]["metric"] == "delivery_days" and plan.direction == "down" and not mismatch_dir:
                try:
                    wi = whatif_tool.simulate(orders, window, 0.20, attr.filters.get("region"))
                    trace.log("Analyst", "what-if simulation (-20% delivery time)",
                              f"elasticity beta={wi['beta']:.3f} (t={wi['beta_t']:.1f}); +{wi['extra_revenue']:,.0f} revenue estimated")
                except Exception as e:
                    trace.log("Analyst", "what-if skipped", str(e))
        return {"attr": attr, "vp": vp, "mechanism": mech, "weekly": weekly, "slice_counts": slice_counts, "whatif": wi,
                "mismatch": bool(mismatch_dir), "orders": orders}

    def _tag_orders(self, window, baseline) -> pd.DataFrame:
        return _tag(self._pull_orders(), baseline, window)

    @staticmethod
    def _weekly(orders: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        o = orders.assign(week=orders["order_date"].dt.to_period("W").dt.start_time, in_slice=mask)
        w = o.groupby(["week", "in_slice"])["delivery_days"].mean().unstack("in_slice")
        w.columns = ["rest" if c is False or c == False else "slice" for c in w.columns]
        return w.reset_index()

    def _facts(self, plan, window, baseline, wl, bl, monthly, st, anomaly_window, checks, confidence) -> dict:
        attr = st["attr"]
        matches = not st["mismatch"]
        facts: dict = {
            "metric": plan.metric, "direction": plan.direction, "window_label": wl, "baseline_label": bl,
            "baseline_total": attr.base_total, "window_total": attr.window_total, "delta": attr.delta, "pct_change": attr.pct,
            "matches_direction": matches, "volume_price": st["vp"], "rest_delta": attr.rest_delta,
            "path": [asdict(s) for s in attr.path], "hypotheses": plan.hypotheses,
        }
        if attr.path:
            last = attr.path[-1]
            facts["top_driver"] = {"label": attr.slice_label, "delta": attr.slice_delta, "share_net": attr.slice_delta / attr.delta if attr.delta else float("nan"),
                                   "filters": attr.filters}
        else:
            facts["top_driver"] = None
        # biggest movers per dimension (used when nothing is concentrated, or the direction doesn't match)
        sign = -1 if plan.direction == "down" else 1
        movers = []
        for d, t in attr.dim_tables.items():
            r = t.iloc[0] if sign < 0 else t.iloc[-1]
            if r["delta"] * sign > 0:
                movers.append({"label": segment_label(d, str(r["segment"])), "delta": float(r["delta"])})
        facts["top_movers"] = sorted(movers, key=lambda x: -abs(x["delta"]))[:3]

        mech = st["mechanism"]
        facts["mechanism"], facts["rating"] = None, None
        if mech is not None and not mech.empty:
            cause = mech[mech["metric"].isin(["delivery_days", "cancelled"]) & mech["significant"]]
            if not cause.empty:
                facts["mechanism"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                      for k, v in cause.iloc[0].to_dict().items() if k != "significant"}
            rt = mech[(mech["metric"] == "rating") & mech["significant"]]
            if not rt.empty and facts["mechanism"]:
                facts["rating"] = {k: float(v) for k, v in rt.iloc[0].to_dict().items() if k in ("slice_base", "slice_window", "slice_change", "z")}

        yoy = None
        ws, we = window
        bs, be = baseline
        py = lambda a, b: monthly[(monthly.index >= a - pd.DateOffset(years=1)) & (monthly.index < b - pd.DateOffset(years=1))].sum()
        if ws - pd.DateOffset(years=1) >= monthly.index.min() and py(ws, we) > 0:
            yoy = {"window": attr.window_total / py(ws, we) - 1, "prior": None}
            if bs - pd.DateOffset(years=1) >= monthly.index.min() and py(bs, be) > 0 and plan.baseline_kind == "prev":
                yoy["prior"] = attr.base_total / py(bs, be) - 1
        facts["yoy"] = yoy if plan.baseline_kind == "prev" else None
        facts["whatif"] = ({k: v for k, v in st["whatif"].items() if k != "caveat"} if st["whatif"] else None)
        facts["anomaly"] = ({"start": str(anomaly_window.start.date()), "months": anomaly_window.months, "t": anomaly_window.severity,
                             "iforest_agreement": anomaly_window.corroborated} if anomaly_window else None)
        facts["confidence"] = confidence
        facts["checks_total"] = sum(c.status in ("pass", "warn", "fail") for c in checks)
        facts["checks_passed"] = sum(c.status == "pass" for c in checks)
        return facts
