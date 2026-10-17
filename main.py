"""FastAPI service.  Run:  uvicorn api.main:app --reload   (docs at /docs)"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from analyst.agent import DataAnalystAgent, InvestigationError
from analyst.bootstrap import ensure_database
from analyst.config import get_settings

app = FastAPI(title="Autonomous AI Data Analyst Agent", version="1.0.0")


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


@app.get("/health")
def health():
    return {"status": "ok", "mode": get_agent().mode}


@app.post("/investigate")
def investigate(q: Question):
    try:
        return get_agent().investigate(q.question).to_dict()
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
