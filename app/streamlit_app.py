"""Streamlit UI for Autonomous AI Data Analyst.
Run: streamlit run app/streamlit_app.py
"""
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyst.agent import DataAnalystAgent, InvestigationError
from analyst.bootstrap import ensure_database
from analyst.config import Settings, get_settings
from analyst.llm import AnthropicLLM, OpenAILLM
from analyst.report import build_figures, render_html, render_markdown

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing database & agent...")
def init_system(provider_override: str | None = None, api_key_override: str | None = None,
                model_override: str | None = None, base_url_override: str | None = None) -> DataAnalystAgent:
    base_settings = get_settings()
    ensure_database(base_settings)

    # Allow session-level overrides from UI
    if provider_override and provider_override != "Auto (from .env)":
        prov = provider_override.lower()
        if "anthropic" in prov:
            settings = Settings(
                provider="anthropic",
                api_key=api_key_override or base_settings.api_key,
                model=model_override or base_settings.model or "claude-sonnet-5-5",
            )
        elif "openai" in prov:
            settings = Settings(
                provider="openai",
                openai_api_key=api_key_override or base_settings.openai_api_key or base_settings.api_key,
                openai_model=model_override or base_settings.openai_model or "gpt-4o-mini",
                openai_base_url=base_url_override or base_settings.openai_base_url,
            )
        elif "ollama" in prov:
            settings = Settings(
                provider="ollama",
                ollama_model=model_override or base_settings.ollama_model or "llama3.2",
                ollama_base_url=base_url_override or base_settings.ollama_base_url,
            )
        elif "offline" in prov:
            settings = Settings(provider="", api_key=None, openai_api_key=None)
        else:
            settings = base_settings
    else:
        settings = base_settings

    return DataAnalystAgent.open(settings)


# --- Sidebar Setup ---
st.sidebar.title("⚙️ Agent Settings")

# Model configuration
with st.sidebar.expander("🤖 Model & Provider", expanded=False):
    provider_choice = st.selectbox(
        "LLM Provider",
        ["Auto (from .env)", "Anthropic (Claude)", "OpenAI (GPT-4o)", "Ollama (Local)", "Offline (Rule-based)"],
        index=0,
    )
    custom_api_key = st.text_input("API Key (optional override)", type="password")
    custom_model = st.text_input("Model Name (optional override)", placeholder="e.g. gpt-4o, claude-3-5-sonnet")
    custom_base_url = st.text_input("Base URL (for Ollama/Groq)", placeholder="e.g. http://localhost:11434/v1")

    if st.button("Apply Model Settings"):
        st.cache_resource.clear()
        st.rerun()

agent = init_system(
    provider_choice if provider_choice != "Auto (from .env)" else None,
    custom_api_key or None,
    custom_model or None,
    custom_base_url or None,
)

# Sidebar Database Info
st.sidebar.subheader("📦 Database Stats")
try:
    tables = agent.db.list_tables()
    order_tbl = next((t for t in tables if t["table_name"] == "orders"), None)
    customer_tbl = next((t for t in tables if t["table_name"] == "customers"), None)
    d_min, d_max = agent.db.date_range()

    st.sidebar.metric("Total Orders", f"{order_tbl['row_count']:,}" if order_tbl else "N/A")
    st.sidebar.metric("Customers", f"{customer_tbl['row_count']:,}" if customer_tbl else "N/A")
    st.sidebar.caption(f"🗓️ Date Range: {d_min.strftime('%b %Y')} - {d_max.strftime('%b %Y')}")
except Exception:
    st.sidebar.caption("Custom / in-memory database active.")

if st.sidebar.button("🔄 Reset / Regenerate Demo Data"):
    with st.spinner("Regenerating demo dataset..."):
        ensure_database(get_settings(), force=True)
        st.cache_resource.clear()
        st.success("Database regenerated!")
        st.rerun()

# --- Main Page Header ---
st.title("📊 Autonomous AI Data Analyst")
status_color = "green" if "LLM" in agent.mode else "orange"
st.markdown(
    f"**Mode:** <span style='color:{status_color}; font-weight:600;'>{agent.mode}</span> &nbsp;|&nbsp; "
    f"**Principle:** LLM plans & narrates; DuckDB + pandas compute; Critic independently verifies.",
    unsafe_allow_html=True,
)
st.write("")

# --- Navigation Tabs ---
tab_inv, tab_ask, tab_what, tab_data = st.tabs([
    "🔎 Root-Cause Investigation",
    "💬 Ask the Data (Text-to-SQL)",
    "🧪 What-If Simulation",
    "📁 Data Explorer & Ingestion",
])

# -----------------------------------------------------------------------------
# TAB 1: INVESTIGATE
# -----------------------------------------------------------------------------
with tab_inv:
    st.subheader("Autonomous Root-Cause Investigation")
    st.caption("Ask why a high-level metric moved. The agent detects anomalies, drills down dimensions, verifies mechanisms, and re-computes everything.")

    examples = [
        "Why did revenue drop in Q3?",
        "Why did revenue fall in Q1 2024?",
        "Why did revenue drop in Q4 2024?",
        "Why did orders drop in Q3 2024 compared to last year?",
    ]

    col_chips = st.columns(len(examples))
    for col, ex in zip(col_chips, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["inv_input"] = ex

    question = st.text_input(
        "Enter your business question:",
        value=st.session_state.get("inv_input", examples[0]),
        key="inv_input_field",
    )

    if st.button("🚀 Investigate Question", type="primary") and question:
        with st.spinner("Executing multi-agent investigation (Planner -> Coder -> Analyst -> Critic -> Narrator)..."):
            try:
                inv = agent.investigate(question)
            except InvestigationError as e:
                st.warning(f"⚠️ Investigation Notice: {e}")
                st.stop()

        confidence_badge = {
            "high": ("🟢 High Confidence", "#2e8b57"),
            "medium": ("🟠 Medium Confidence", "#d98e04"),
            "low": ("🔴 Low Confidence", "#d1495b"),
        }.get(inv.confidence, (inv.confidence, "#6b7280"))

        st.markdown(
            f"<div style='display:inline-block; background-color:{confidence_badge[1]}; color:white; "
            f"padding:4px 12px; border-radius:16px; font-weight:600; margin-bottom:12px;'>"
            f"{confidence_badge[0]}</div>",
            unsafe_allow_html=True,
        )

        st.info(inv.narrative)

        # Figures
        figs = build_figures(inv)
        c1, c2 = st.columns(2)
        if "timeline" in figs:
            c1.plotly_chart(figs["timeline"], use_container_width=True)
        if "bridge" in figs:
            c2.plotly_chart(figs["bridge"], use_container_width=True)

        c3, c4 = st.columns(2)
        if "segments" in figs:
            c3.plotly_chart(figs["segments"], use_container_width=True)
        if "mechanism" in figs:
            c4.plotly_chart(figs["mechanism"], use_container_width=True)

        # Checks & Trace
        col_chk, col_trc = st.columns(2)
        with col_chk:
            with st.expander("✅ Independent Critic Verification Checks", expanded=True):
                icon_map = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
                for c in inv.checks:
                    st.markdown(f"{icon_map.get(c.status, '•')} **{c.name}**  \n<small>{c.detail}</small>", unsafe_allow_html=True)

        with col_trc:
            with st.expander("🧠 Agent Execution Trace", expanded=False):
                for e in inv.trace.to_list():
                    st.markdown(
                        f"**`{e['t']}s` {e['agent']}** - *{e['action']}*"
                        + (f"<br><small>{e['detail']}</small>" if e["detail"] else ""),
                        unsafe_allow_html=True,
                    )
                    if e["sql"]:
                        st.code(e["sql"], language="sql")

        # Report Downloads
        st.write("---")
        d1, d2 = st.columns([1, 1])
        d1.download_button(
            "📥 Download Interactive HTML Report",
            render_html(inv),
            file_name="investigation_report.html",
            mime="text/html",
            use_container_width=True,
        )
        d2.download_button(
            "📄 Download Markdown Summary",
            render_markdown(inv),
            file_name="investigation_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# TAB 2: ASK THE DATA
# -----------------------------------------------------------------------------
with tab_ask:
    st.subheader("Ad-Hoc Text-to-SQL Querying")
    st.caption("Ask natural language questions. The Coder agent writes SQL, self-repairs errors, executes deterministically, and presents the data.")

    ask_presets = [
        "Top 3 categories by revenue in Q3 2024",
        "Total revenue and orders by region in 2024",
        "Average basket size and units by customer type",
        "Monthly revenue trend in 2024",
    ]
    p_cols = st.columns(len(ask_presets))
    for col, pre in zip(p_cols, ask_presets):
        if col.button(pre, use_container_width=True):
            st.session_state["ask_input"] = pre

    q = st.text_input(
        "Ask anything about your data:",
        value=st.session_state.get("ask_input", ask_presets[0]),
        key="ask_input_field",
    )

    if st.button("🔍 Run Query", type="primary") and q:
        with st.spinner("Synthesizing SQL and executing query..."):
            res = agent.ask(q)

        if res.sql:
            st.code(res.sql, language="sql")

        if res.df is not None and not res.df.empty:
            st.dataframe(res.df, use_container_width=True)

            # Auto-visualization: if 1 text column and 1 numeric column -> bar/line chart
            cols = res.df.columns
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(res.df[c])]
            cat_cols = [c for c in cols if c not in num_cols]

            if len(num_cols) >= 1 and len(cat_cols) >= 1:
                with st.expander("📈 Auto-Generated Chart", expanded=True):
                    x_c = cat_cols[0]
                    y_c = num_cols[0]
                    if len(res.df) <= 30:
                        fig = px.bar(res.df, x=x_c, y=y_c, title=f"{y_c} by {x_c}", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)

            # CSV Download
            csv_data = res.df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", csv_data, "query_results.csv", "text/csv")

        st.success(res.answer)
        for wmsg in res.warnings:
            st.warning(f"⚠️ {wmsg}")

# -----------------------------------------------------------------------------
# TAB 3: WHAT-IF SIMULATION
# -----------------------------------------------------------------------------
with tab_what:
    st.subheader("Econometric What-If Simulation")
    st.caption("Estimate repeat order uplift from counterfactual delivery speedups using a two-way fixed effects panel model.")

    col_reg, col_red = st.columns(2)
    with col_reg:
        region = st.selectbox("Select Target Region", ["West", "North", "South", "East", "Central", "All regions"])
    with col_red:
        red = st.slider("Delivery Time Reduction", 5, 50, 20, step=5, format="%d%%") / 100

    if st.button("🧪 Run What-If Simulation", type="primary"):
        try:
            out = agent.whatif(red, None if region == "All regions" else region)
        except InvestigationError as e:
            st.warning(str(e))
            st.stop()

        cur = agent.settings.currency
        m1, m2, m3 = st.columns(3)
        m1.metric("Projected Extra Repeat Orders", f"+{out['extra_repeat_orders']:,.0f}")
        m2.metric("Projected Extra Revenue", f"+{cur}{out['extra_revenue']:,.0f}")
        m3.metric("Average Delivery Time", f"{out['avg_days_before']:.1f}d → {out['avg_days_after']:.1f}d", f"-{red:.0%}")

        # Visualization of scenario
        fig_wi = go.Figure()
        fig_wi.add_trace(go.Bar(
            name="Baseline",
            x=["Delivery Days", "Estimated Revenue ($10K)"],
            y=[out["avg_days_before"], 0],
            marker_color="#2b6cb0",
        ))
        fig_wi.add_trace(go.Bar(
            name="Simulated Scenario",
            x=["Delivery Days", "Estimated Revenue ($10K)"],
            y=[out["avg_days_after"], out["extra_revenue"] / 10000],
            marker_color="#2e8b57",
        ))
        fig_wi.update_layout(barmode="group", height=320, title="Baseline vs Counterfactual Simulation")
        st.plotly_chart(fig_wi, use_container_width=True)

        st.caption(
            f"Confidence Range: {cur}{out['revenue_low']:,.0f} to {cur}{out['revenue_high']:,.0f} · "
            f"Elasticity Beta: {out['beta']:.3f} per day (t={out['beta_t']:.1f}) · Period: {out['window_label']}"
        )
        if out.get("extrapolation"):
            st.warning("⚠️ Warning: Scenario exceeds historically observed delivery improvements (extrapolation).")
        st.info(f"ℹ️ {out['caveat']}")

# -----------------------------------------------------------------------------
# TAB 4: DATA EXPLORER & INGESTION
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Data Explorer & Dynamic CSV Ingestion")
    st.caption("Inspect live tables, schemas, and register custom datasets for real-world analysis.")

    exp_col1, exp_col2 = st.columns([1, 1])

    with exp_col1:
        st.markdown("### 📋 Database Tables")
        try:
            tbl_list = agent.db.list_tables()
            tbl_df = pd.DataFrame(tbl_list)
            st.dataframe(tbl_df, use_container_width=True)

            selected_tbl = st.selectbox("Inspect Table / View", [t["table_name"] for t in tbl_list])
            if selected_tbl:
                schema_cols = agent.db.table_schema(selected_tbl)
                st.markdown(f"**Schema for `{selected_tbl}`:**")
                st.dataframe(pd.DataFrame(schema_cols), use_container_width=True)

                st.markdown(f"**Sample Rows (10 rows):**")
                sample_df = agent.db.sample_rows(selected_tbl, limit=10)
                st.dataframe(sample_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading tables: {e}")

    with exp_col2:
        st.markdown("### 📤 Upload Custom CSV Dataset")
        st.caption("Upload a CSV file to create or register a new queryable table in DuckDB.")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        custom_tbl_name = st.text_input("Table Name for Dataset", placeholder="e.g. quarterly_sales")

        if uploaded_file and st.button("📥 Load CSV into Database", type="primary"):
            try:
                raw_name = custom_tbl_name or uploaded_file.name.rsplit(".", 1)[0]
                clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", raw_name).lower()
                df_upload = pd.read_csv(uploaded_file)
                agent.db.register_dataframe(clean_name, df_upload)
                st.success(f"Successfully registered table `{clean_name}` with {len(df_upload):,} rows and {len(df_upload.columns)} columns!")
                st.dataframe(df_upload.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to ingest CSV: {e}")

        st.markdown("### ⚡ Direct Safe SQL Query")
        custom_sql = st.text_area(
            "Execute read-only SQL:",
            "SELECT region, COUNT(*) AS total_orders, ROUND(SUM(order_revenue), 0) AS total_revenue FROM v_orders GROUP BY 1 ORDER BY 3 DESC",
            height=100,
        )
        if st.button("Run SQL Query"):
            try:
                res_df = agent.db.query(custom_sql, limit=100)
                st.dataframe(res_df, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")
