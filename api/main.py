"""FastAPI service.  Run:  uvicorn api.main:app --reload   (docs at /docs)"""
from __future__ import annotations

import io
import re
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analyst.agent import DataAnalystAgent, InvestigationError
from analyst.bootstrap import ensure_database
from analyst.config import get_settings
from analyst.report import render_html, render_markdown

app = FastAPI(
    title="Autonomous AI Data Analyst Agent",
    description="Production-grade AI Data Analyst API with multi-agent root cause analysis, text-to-SQL, and what-if simulation.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_agent() -> DataAnalystAgent:
    settings = get_settings()
    ensure_database(settings)
    return DataAnalystAgent.open(settings)


class Question(BaseModel):
    question: str = Field(..., min_length=3, examples=["Why did revenue drop in Q3?"])


class WhatIf(BaseModel):
    reduction: float = Field(0.20, gt=0, lt=0.9)
    region: str | None = None


class SafeSQLQuery(BaseModel):
    sql: str = Field(..., min_length=5, examples=["SELECT region, SUM(order_revenue) FROM v_orders GROUP BY 1"])
    limit: int = Field(500, gt=0, le=5000)


@app.get("/")
def root():
    agent = get_agent()
    return {
        "service": "Autonomous AI Data Analyst API",
        "version": "1.1.0",
        "mode": agent.mode,
        "docs_url": "/docs",
        "endpoints": [
            "/health",
            "/tables",
            "/investigate",
            "/investigate/report",
            "/ask",
            "/whatif",
            "/query",
            "/upload",
        ],
    }


@app.get("/health")
def health():
    agent = get_agent()
    return {
        "status": "ok",
        "mode": agent.mode,
        "database": str(agent.settings.db_path),
        "currency": agent.settings.currency,
    }


@app.get("/tables")
def list_tables():
    agent = get_agent()
    return {"tables": agent.db.list_tables()}


@app.get("/tables/{table_name}/schema")
def get_table_schema(table_name: str):
    agent = get_agent()
    try:
        schema = agent.db.table_schema(table_name)
        return {"table_name": table_name, "columns": schema}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tables/{table_name}/sample")
def get_table_sample(table_name: str, limit: int = 10):
    agent = get_agent()
    try:
        df = agent.db.sample_rows(table_name, limit=limit)
        return {"table_name": table_name, "columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/investigate")
def investigate(q: Question):
    try:
        return get_agent().investigate(q.question).to_dict()
    except InvestigationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/investigate/report")
def generate_report(q: Question, format: str = "html"):
    try:
        inv = get_agent().investigate(q.question)
        if format.lower() == "markdown":
            return {"format": "markdown", "content": render_markdown(inv)}
        return {"format": "html", "content": render_html(inv)}
    except InvestigationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
def ask(q: Question):
    return get_agent().ask(q.question).to_dict()


@app.post("/whatif")
def whatif(w: WhatIf):
    try:
        return get_agent().whatif(w.reduction, w.region)
    except InvestigationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
def execute_query(q: SafeSQLQuery):
    agent = get_agent()
    try:
        df = agent.db.query(q.sql, limit=q.limit)
        truncated = df.attrs.get("truncated", False)
        return {
            "sql": q.sql,
            "row_count": len(df),
            "truncated": truncated,
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), table_name: str | None = None):
    agent = get_agent()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    clean_name = table_name or file.filename.rsplit(".", 1)[0]
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name).lower()
    if not clean_name:
        clean_name = "uploaded_data"

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        agent.db.register_dataframe(clean_name, df)
        return {
            "status": "success",
            "table_name": clean_name,
            "rows": len(df),
            "columns": list(df.columns),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {e}")
