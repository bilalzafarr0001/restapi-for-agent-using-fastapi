from fastapi.testclient import TestClient

from app import tools
from app.main import app

client = TestClient(app)


def test_list_agents():
    response = client.get("/agents")
    assert response.json() == ["search", "news", "math"]


def test_unknown_agent_returns_404():
    response = client.post("/agents/cooking/chat", json={"message": "hi"})
    assert response.status_code == 404


def test_math_tools():
    assert tools.add.invoke({"a": 2, "b": 3}) == 5
    assert tools.subtract.invoke({"a": 10, "b": 4}) == 6
    assert tools.multiply.invoke({"a": 2450, "b": 0.15}) == 367.5
    assert tools.divide.invoke({"a": 1, "b": 0}) == "Error: cannot divide by zero."


def test_web_search(monkeypatch):
    # Replace the real Serper call with a fake one, so the test needs no internet.
    def fake_call_serper(url, payload):
        return {"organic": [{"title": "FastAPI", "link": "https://fastapi.tiangolo.com", "snippet": "A web framework"}]}

    monkeypatch.setattr(tools, "call_serper", fake_call_serper)
    result = tools.web_search.invoke({"query": "fastapi"})
    assert "https://fastapi.tiangolo.com" in result
