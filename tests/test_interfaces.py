from fastapi.testclient import TestClient


def test_api_endpoints(monkeypatch, agent):
    import api.main as m
    monkeypatch.setattr(m, "get_agent", lambda: agent)
    c = TestClient(m.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/investigate", json={"question": "Why did revenue drop in Q3?"}).json()
    assert r["confidence"] == "high" and r["facts"]["top_driver"]["filters"]["region"] == "West"
    assert c.post("/investigate", json={"question": "Why did revenue drop in Q1 2025?"}).status_code == 400
    a = c.post("/ask", json={"question": "Top 3 categories by revenue in Q3 2024"}).json()
    assert a["mode"] == "rules" and len(a["rows"]) == 3
    w = c.post("/whatif", json={"reduction": 0.2, "region": "West"}).json()
    assert w["extra_revenue"] > 0
    assert c.post("/whatif", json={"reduction": 5}).status_code == 422


def test_report_files(q3, tmp_path):
    from analyst.report import save_report
    paths = save_report(q3, tmp_path)
    html = paths["html"].read_text()
    assert "plotly" in html and "confidence: high" in html and "West" in html
    assert "Agent trace" in paths["markdown"].read_text() or "Agent trace" in html
