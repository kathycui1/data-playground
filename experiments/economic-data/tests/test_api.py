from __future__ import annotations

from fastapi.testclient import TestClient

from api import app, get_packet
from compute import build_facts
from tests.conftest import make_panel


def test_health_and_facts_override():
    packet = build_facts(make_panel())
    app.dependency_overrides[get_packet] = lambda: packet
    client = TestClient(app)
    try:
        assert client.get("/health").json() == {"status": "ok"}
        payload = client.get("/facts").json()
        assert payload["world"]["iso"] == "WLD"
        assert len(payload["focus"]) == 6
        briefing = client.get("/briefing", params={"skip_llm": True}).json()
        assert briefing["skipped_reason"]
    finally:
        app.dependency_overrides.clear()


def test_ask_without_key_is_skipped(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    packet = build_facts(make_panel())
    app.dependency_overrides[get_packet] = lambda: packet
    client = TestClient(app)
    try:
        response = client.post("/ask", json={"question": "Which country has higher inflation?"})
        assert response.status_code == 200
        body = response.json()
        assert body["question"].startswith("Which country")
        assert body["skipped_reason"] == "OPENAI_API_KEY is not set"
    finally:
        app.dependency_overrides.clear()
