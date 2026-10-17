import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analyst.cli import main as cli_main
from analyst.config import Settings
from analyst.db import Database
from analyst.llm import AnthropicLLM, OpenAILLM, get_llm
import api.main as api_module


def test_openai_llm_complete(monkeypatch):
    mock_post = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello from OpenAI!"}}]
    }
    mock_post.return_value = mock_response

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return mock_post(*args, **kwargs)

    monkeypatch.setattr("httpx.Client", MockClient)

    llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini", base_url="https://api.openai.com/v1")
    reply = llm.complete(system="You are an analyst", user="Hello")
    assert reply == "Hello from OpenAI!"
    assert mock_post.call_count == 1
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
    assert call_args[1]["json"]["model"] == "gpt-4o-mini"


def test_get_llm_provider_selection():
    # 1. Offline (no keys)
    s_offline = Settings(provider="", api_key=None, openai_api_key=None)
    assert get_llm(s_offline) is None

    # 2. OpenAI provider
    s_openai = Settings(provider="openai", openai_api_key="sk-openai-key")
    llm_openai = get_llm(s_openai)
    assert isinstance(llm_openai, OpenAILLM)
    assert llm_openai.api_key == "sk-openai-key"

    # 3. Ollama provider
    s_ollama = Settings(provider="ollama", ollama_base_url="http://localhost:11434/v1", ollama_model="llama3.2")
    llm_ollama = get_llm(s_ollama)
    assert isinstance(llm_ollama, OpenAILLM)
    assert llm_ollama.base_url == "http://localhost:11434/v1"

    # 4. Anthropic provider
    s_anthropic = Settings(provider="anthropic", api_key="sk-ant-test")
    llm_anthropic = get_llm(s_anthropic)
    assert isinstance(llm_anthropic, AnthropicLLM)


def test_database_inspection_and_registration(db):
    # Test list_tables
    tables = db.list_tables()
    names = [t["table_name"] for t in tables]
    assert "orders" in names and "v_orders" in names
    orders_info = next(t for t in tables if t["table_name"] == "orders")
    assert orders_info["row_count"] > 1000

    # Test table_schema
    schema = db.table_schema("v_orders")
    col_names = [c["column_name"] for c in schema]
    assert "order_id" in col_names and "customer_type" in col_names

    # Test sample_rows
    sample = db.sample_rows("v_orders", limit=3)
    assert len(sample) == 3

    # Test registering dynamic DataFrame
    custom_df = pd.DataFrame({"product": ["A", "B", "C"], "sales": [100, 200, 300]})
    db.register_dataframe("custom_inventory", custom_df)
    res = db.query("SELECT SUM(sales) AS total FROM custom_inventory")
    assert res.iloc[0, 0] == 600


def test_api_real_world_endpoints(monkeypatch, agent):
    monkeypatch.setattr(api_module, "get_agent", lambda: agent)
    client = TestClient(api_module.app)

    # Test root endpoint
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "Autonomous AI Data Analyst API" in root_resp.json()["service"]

    # Test CORS header
    cors_resp = client.options("/", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert cors_resp.status_code == 200

    # Test tables endpoint
    t_resp = client.get("/tables")
    assert t_resp.status_code == 200
    table_names = [t["table_name"] for t in t_resp.json()["tables"]]
    assert "v_sales" in table_names

    # Test table schema
    s_resp = client.get("/tables/v_sales/schema")
    assert s_resp.status_code == 200
    assert any(c["column_name"] == "revenue" for c in s_resp.json()["columns"])

    # Test table sample
    sample_resp = client.get("/tables/v_sales/sample?limit=5")
    assert sample_resp.status_code == 200
    assert len(sample_resp.json()["rows"]) == 5

    # Test direct safe query endpoint
    q_resp = client.post("/query", json={"sql": "SELECT category, ROUND(SUM(revenue), 0) AS rev FROM v_sales GROUP BY 1", "limit": 10})
    assert q_resp.status_code == 200
    assert len(q_resp.json()["rows"]) > 0

    # Test report generation endpoint
    r_resp = client.post("/investigate/report?format=markdown", json={"question": "Why did revenue drop in Q3?"})
    assert r_resp.status_code == 200
    assert "Investigation:" in r_resp.json()["content"]

    # Test CSV upload endpoint
    csv_content = b"user_id,amount\n1,50.5\n2,99.9\n"
    files = {"file": ("test_upload.csv", io.BytesIO(csv_content), "text/csv")}
    up_resp = client.post("/upload", files=files)
    assert up_resp.status_code == 200
    assert up_resp.json()["table_name"] == "test_upload"
    assert up_resp.json()["rows"] == 2


def test_cli_tables_and_query(capsys):
    ret = cli_main(["tables"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "v_orders" in captured.out

    ret_schema = cli_main(["tables", "--schema", "v_orders"])
    assert ret_schema == 0
    captured_schema = capsys.readouterr()
    assert "delivery_days" in captured_schema.out

    ret_query = cli_main(["query", "SELECT 42 AS answer"])
    assert ret_query == 0
    captured_query = capsys.readouterr()
    assert "42" in captured_query.out
