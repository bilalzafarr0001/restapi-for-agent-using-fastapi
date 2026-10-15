"""Tools: Python functions that the agents can call.

@tool shows the LLM each function's name, inputs and docstring.
If a tool fails, agents.py gives the error to the LLM, so tools don't handle errors themselves.
"""

import os
from datetime import date

import httpx
from langchain.tools import tool


def call_serper(url: str, payload: dict) -> dict:
    """Call the Serper API (Google search). Needs SERPER_API_KEY in .env."""
    response = httpx.post(url, json=payload, headers={"X-API-KEY": os.environ["SERPER_API_KEY"]}, timeout=30)
    response.raise_for_status()  # Stop with an error if the call failed (for example, a wrong API key).
    return response.json()


def format_results(results: list[dict]) -> str:
    """Turn search results into text: a title, link and snippet for each one."""
    text = f"Today is {date.today()}.\n\n"  # Helps the LLM judge how recent a result is.
    for result in results[:5]:
        text += f"{result['title']}\n{result['link']}\n{result.get('snippet', '')}\n\n"
    return text


# --- Search tools (Serper) ---


@tool
def web_search(query: str) -> str:
    """Search Google. Returns the top results with their links."""
    print(f"[TOOL] web_search: {query}")
    data = call_serper("https://google.serper.dev/search", {"q": query})
    return format_results(data.get("organic", []))


@tool
def news_search(query: str) -> str:
    """Search Google News. Returns recent articles with their links."""
    print(f"[TOOL] news_search: {query}")
    data = call_serper("https://google.serper.dev/news", {"q": query})
    return format_results(data.get("news", []))


@tool
def read_webpage(url: str) -> str:
    """Open a web page from its link and return its text."""
    print(f"[TOOL] read_webpage: {url}")
    data = call_serper("https://scrape.serper.dev", {"url": url})
    return data.get("text", "")[:3000]  # Only the start of the page, to stay within Groq's free limit.


@tool
def places_search(query: str) -> str:
    """Search Google Maps for places, for example "biryani restaurants in Lahore"."""
    print(f"[TOOL] places_search: {query}")
    data = call_serper("https://google.serper.dev/places", {"q": query})
    text = ""
    for place in data.get("places", [])[:5]:
        text += f"{place['title']}, {place.get('address')}, rating {place.get('rating')} ({place.get('ratingCount')} reviews), "
        text += f"phone {place.get('phoneNumber')}, website {place.get('website')}\n"
    return text or "No places found."


# --- Weather tool (WeatherAPI.com) ---


@tool
def get_weather(city: str) -> str:
    """Get the current temperature and weather condition in a city, for example "Lahore"."""
    print(f"[TOOL] get_weather: {city}")
    params = {"key": os.environ["WEATHER_API_KEY"], "q": city}
    data = httpx.get("https://api.weatherapi.com/v1/current.json", params=params).json()
    if "error" in data:
        return f"Could not get the weather for {city}: {data['error']['message']}"  # e.g. "No matching location found."

    place, now = data["location"], data["current"]
    return f"Right now in {place['name']}, {place['country']}: {now['temp_c']}°C, {now['condition']['text']}."


# --- Math tools ---


@tool
def add(a: float, b: float) -> float:
    """Add two numbers: a + b."""
    print(f"[TOOL] add: {a} + {b}")
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers: a - b."""
    print(f"[TOOL] subtract: {a} - {b}")
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers: a * b."""
    print(f"[TOOL] multiply: {a} * {b}")
    return a * b


@tool
def divide(a: float, b: float) -> float | str:
    """Divide two numbers: a / b."""
    print(f"[TOOL] divide: {a} / {b}")
    if b == 0:
        return "Error: cannot divide by zero."
    return a / b
