import copy

import pandas as pd
import pytest

from analyst.agent import InvestigationError
from analyst.agents import Narrator, ungrounded_numbers
from analyst.critic import Critic
from analyst.tools import whatif
from analyst.tools.drivers import ITEM_DIMS, attribute

from .conftest import FakeLLM


# ----------------------------------------------------------- end-to-end root cause
def test_finds_injected_root_cause(q3):
    f = q3.facts
    assert f["window_label"] == "Q3 2024" and f["matches_direction"]
    assert f["pct_change"] < -0.08
    assert f["top_driver"]["filters"] == {"region": "West", "customer_type": "repeat"}
    assert f["mechanism"]["metric"] == "delivery_days"
    assert 2.8 < f["mechanism"]["slice_change"] < 3.8          # injected +3.3 days
    assert f["volume_price"]["volume_effect"] < f["volume_price"]["price_effect"]  # a volume story, not a price story
    assert "West" in q3.narrative and "delivery" in q3.narrative


def test_all_checks_pass_and_confidence_high(q3):
    assert not [c for c in q3.checks if c.status == "fail"]
    assert q3.confidence == "high"
    assert q3.facts["checks_passed"] == q3.facts["checks_total"]


def test_trace_records_agents(q3):
    agents = {e["agent"] for e in q3.trace.to_list()}
    assert {"Planner", "Coder", "Analyst", "Critic", "Narrator"} <= agents
    assert any(e["sql"] for e in q3.trace.to_list())


def test_no_period_uses_detector(agent):
    assert agent.investigate("Why did revenue drop recently?").facts["window_label"] == "Q3 2024"


def test_refuses_to_invent_a_drop(agent):
    inv = agent.investigate("Why did revenue drop in Q4 2024?")
    assert not inv.facts["matches_direction"] and inv.facts["top_driver"] is None
    assert "did not fall" in inv.narrative


def test_seasonal_decline_is_broad_based(agent):
    inv = agent.investigate("Why did revenue fall in Q1 2024?")
    assert inv.facts["matches_direction"] and inv.facts["top_driver"] is None
    assert "broad-based" in inv.narrative and inv.facts["yoy"]["window"] > 0


def test_out_of_range_period_errors_cleanly(agent):
    with pytest.raises(InvestigationError):
        agent.investigate("Why did revenue drop in Q1 2025?")


def test_yoy_orders_variant(agent):
    inv = agent.investigate("why did orders drop in Q3 2024 compared to last year")
    assert inv.facts["metric"] == "orders" and inv.facts["baseline_label"] == "Q3 2023"
    assert inv.facts["top_driver"]["filters"].get("region") == "West"


# ----------------------------------------------------------- critic must catch tampering
def test_critic_catches_wrong_numbers(agent, q3):
    d = q3.data
    bad = copy.deepcopy(d["attr"])
    bad.window_total *= 1.01                         # silently corrupt a headline number
    checks = agent.critic.verify(metric="revenue", window=d["window"], baseline=d["baseline"], attr=bad, monthly=d["monthly"],
                                 slice_counts={"base": 100, "window": 100}, mechanism=d["mechanism"], anomaly_window=None)
    failed = {c.name for c in checks if c.status == "fail"}
    assert "Totals match raw-table SQL" in failed and "Totals match monthly series" in failed
    assert Critic.confidence(checks) == "low"


def test_critic_flags_small_samples(agent, q3):
    d = q3.data
    checks = agent.critic.verify(metric="revenue", window=d["window"], baseline=d["baseline"], attr=d["attr"], monthly=d["monthly"],
                                 slice_counts={"base": 5, "window": 4}, mechanism=d["mechanism"], anomaly_window=None)
    assert Critic.confidence(checks) == "medium"


def test_drilldown_ignores_proportional_noise():
    # every segment shrinks by the same 20%: nothing should be blamed
    df = pd.DataFrame({"period": ["base", "window"] * 4, "region": list("AABBCCDD"), "x": ["u"] * 8,
                       "revenue": [100, 80, 200, 160, 300, 240, 400, 320]})
    a = attribute(df, "revenue", ["region"], "down")
    assert a.path == [] and abs(a.pct + 0.2) < 1e-9


# ----------------------------------------------------------- what-if
def test_whatif_elasticity_and_uplift(agent):
    out = agent.whatif(0.20, "West")
    assert out["beta"] < -0.15 and abs(out["beta_t"]) > 10
    assert out["extra_revenue"] > 0 and out["revenue_low"] <= out["extra_revenue"] <= out["revenue_high"]
    assert out["window_label"] == "Q3 2024"
    bigger = agent.whatif(0.40, "West")
    assert bigger["extra_revenue"] > out["extra_revenue"]


# ----------------------------------------------------------- narrator grounding
def test_grounding_check():
    facts = {"pct_change": -0.1032, "window_total": 997137.0, "delta": -114741.0}
    assert ungrounded_numbers("Revenue fell 10.3% to $997.1K (-$114.7K) in Q3 2024.", facts) == []
    assert ungrounded_numbers("Revenue fell 25% to $2.1M.", facts) == ["25%", "$2.1M"]


def test_narrator_rejects_hallucinated_numbers(q3):
    from analyst.utils import Trace
    nar = Narrator(FakeLLM(["**Revenue crashed 47%** because of a $9.9M loss."]), "$")
    t = Trace()
    text = nar.write(q3.facts, t)
    assert "47%" not in text and "Revenue fell 10.3%" in text
    assert any("ungrounded" in e.action for e in t.events)


def test_narrator_accepts_grounded_llm_text(q3):
    from analyst.utils import Trace
    ok = "**Revenue fell 10.3% in Q3 2024 vs Q2 2024**, driven by repeat customers in the West region."
    assert Narrator(FakeLLM([ok]), "$").write(q3.facts, Trace()) == ok
