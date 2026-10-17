"""Streamlit UI.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from analyst.agent import DataAnalystAgent, InvestigationError
from analyst.bootstrap import ensure_database
from analyst.config import get_settings
from analyst.report import build_figures, render_html

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Preparing data and agent...")
def load_agent() -> DataAnalystAgent:
    settings = get_settings()
    ensure_database(settings)
    return DataAnalystAgent.open(settings)


agent = load_agent()
st.title("📊 Autonomous AI Data Analyst")
st.caption(f"Mode: {agent.mode}. Numbers come from SQL + pandas and are re-verified by a Critic agent; the LLM only plans and narrates.")

tab_inv, tab_ask, tab_what = st.tabs(["🔎 Investigate", "💬 Ask the data", "🧪 What-if"])

with tab_inv:
    examples = ["Why did revenue drop in Q3?", "Why did revenue fall in Q1 2024?", "Why did revenue drop in Q4 2024?",
                "Why did orders drop in Q3 2024 compared to last year?"]
    if "inv_q" not in st.session_state:
        st.session_state["inv_q"] = examples[0]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, width="stretch"):
            st.session_state["inv_q"] = ex
    question = st.text_input("Business question", key="inv_q")
    if st.button("Investigate", type="primary") and question:
        with st.spinner("Planning, querying, analysing, verifying..."):
            try:
                inv = agent.investigate(question)
            except InvestigationError as e:
                st.warning(str(e))
                st.stop()
        badge = {"high": "🟢", "medium": "🟠", "low": "🔴"}[inv.confidence]
        st.markdown(f"### Executive summary  {badge} confidence: **{inv.confidence}**")
        st.markdown(inv.narrative)
        figs = build_figures(inv)
        c1, c2 = st.columns(2)
        c1.plotly_chart(figs["timeline"], width="stretch")
        c2.plotly_chart(figs["bridge"], width="stretch")
        c3, c4 = st.columns(2)
        c3.plotly_chart(figs["segments"], width="stretch")
        if "mechanism" in figs:
            c4.plotly_chart(figs["mechanism"], width="stretch")
        with st.expander("✅ Verification checks", expanded=True):
            icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
            for c in inv.checks:
                st.write(f"{icon[c.status]} **{c.name}** - {c.detail}")
        with st.expander("🧠 Agent trace (every step and SQL)"):
            for e in inv.trace.to_list():
                st.markdown(f"`{e['t']}s` **{e['agent']}** - {e['action']}" + (f"  \n<small>{e['detail'][:300]}</small>" if e["detail"] else ""), unsafe_allow_html=True)
                if e["sql"]:
                    st.code(e["sql"], language="sql")
        st.download_button("Download HTML report", render_html(inv), file_name="investigation.html", mime="text/html")

with tab_ask:
    q = st.text_input("Ask anything about the data", "Top 3 categories by revenue in Q3 2024", key="ask_q")
    if st.button("Run query") and q:
        res = agent.ask(q)
        if res.sql:
            st.code(res.sql, language="sql")
        if res.df is not None:
            st.dataframe(res.df, width="stretch")
        st.info(res.answer)
        for wmsg in res.warnings:
            st.warning(wmsg)

with tab_what:
    region = st.selectbox("Region", ["West", "North", "South", "East", "Central", "All regions"])
    red = st.slider("Cut in delivery time", 5, 50, 20, step=5, format="%d%%") / 100
    if st.button("Simulate"):
        try:
            out = agent.whatif(red, None if region == "All regions" else region)
        except InvestigationError as e:
            st.warning(str(e))
            st.stop()
        cur = get_settings().currency
        m1, m2, m3 = st.columns(3)
        m1.metric("Extra repeat orders", f"{out['extra_repeat_orders']:,.0f}")
        m2.metric("Extra revenue (estimate)", f"{cur}{out['extra_revenue']:,.0f}")
        m3.metric("Delivery days", f"{out['avg_days_before']:.1f} → {out['avg_days_after']:.1f}")
        st.caption(f"Range {cur}{out['revenue_low']:,.0f} to {cur}{out['revenue_high']:,.0f} · elasticity {out['beta']:.3f} per day (t={out['beta_t']:.1f}) · window {out['window_label']}")
        if out["extrapolation"]:
            st.warning("Scenario goes below delivery times seen in the data: extrapolation.")
        st.caption(out["caveat"])
