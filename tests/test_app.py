import httpx
from fastapi.testclient import TestClient
from langchain.messages import AIMessageChunk, ToolMessage

from app import tools
from app.agents import AGENTS
from app.main import app

client = TestClient(app)


def test_list_agents():
    response = client.get("/agents")
    assert response.json() == ["search", "news", "math", "places", "weather", "assistant"]


def test_chat_endpoint_exists():
    # POST /chat needs no agent name. (Calling it would use the real LLM, so we only check it exists.)
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/chat"]


def test_unknown_agent_returns_404():
    response = client.post("/agents/cooking/chat", json={"message": "hi"})
    assert response.status_code == 404


def test_chat_stream(monkeypatch):
    # A fake agent that streams its answer in pieces, so the test needs no LLM.
    class FakeAgent:
        def stream(self, input, config, stream_mode):
            yield AIMessageChunk(content="Hello "), {}
            yield ToolMessage(content="raw search results", tool_call_id="1"), {}  # Must not be sent.
            yield AIMessageChunk(content="there!"), {}

    monkeypatch.setitem(AGENTS, "fake", FakeAgent())
    response = client.post("/agents/fake/chat/stream", json={"message": "hi"})
    assert response.text == "Hello there!"
    assert response.headers["X-Thread-Id"]


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


def test_places_search(monkeypatch):
    # A fake Serper reply. The second place has no phone or website, like many real places.
    def fake_call_serper(url, payload):
        return {
            "places": [
                {"title": "Waqas Biryani House", "address": "Hall Rd, Lahore", "rating": 4.3,
                 "ratingCount": 14000, "category": "Biryani", "phoneNumber": "0300 1234567"},
                {"title": "Daleem e Khas Matka Biryani", "address": "Raja Center, Lahore",
                 "rating": 4.1, "ratingCount": 699, "category": "Biryani"},
            ]
        }

    monkeypatch.setattr(tools, "call_serper", fake_call_serper)
    result = tools.places_search.invoke({"query": "biryani in Lahore"})
    assert "Waqas Biryani House, Hall Rd, Lahore, rating 4.3 (14000 reviews), phone 0300 1234567" in result
    assert "Daleem e Khas Matka Biryani" in result


def test_get_weather(monkeypatch):
    # A fake WeatherAPI reply, so the test needs no internet.
    fake_reply = {
        "location": {"name": "Lahore", "country": "Pakistan"},
        "current": {"temp_c": 27.1, "condition": {"text": "Partly cloudy"}},
    }
    monkeypatch.setattr(tools.httpx, "get", lambda url, params: httpx.Response(200, json=fake_reply))
    result = tools.get_weather.invoke({"city": "Lahore"})
    assert result == "Right now in Lahore, Pakistan: 27.1°C, Partly cloudy."


def test_get_weather_unknown_city(monkeypatch):
    fake_reply = {"error": {"code": 1006, "message": "No matching location found."}}
    monkeypatch.setattr(tools.httpx, "get", lambda url, params: httpx.Response(400, json=fake_reply))
    result = tools.get_weather.invoke({"city": "Nowhere"})
    assert result == "Could not get the weather for Nowhere: No matching location found."
