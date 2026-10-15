"""Tools: normal Python functions that an agent can choose to call.

The @tool decorator shows the LLM each tool's name, inputs and docstring.
The docstring matters: it is how the LLM knows what the tool does.
"""

import os
from datetime import date

import httpx
from langchain.tools import tool

# ---------------------------------------------------------------------------
# Google search tools (they use the Serper API: https://serper.dev)
# ---------------------------------------------------------------------------


def call_serper(url: str, payload: dict) -> dict:
    """Send a request to Serper and return its JSON reply."""
    response = httpx.post(
        url,
        json=payload,
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
        timeout=30,
    )
    response.raise_for_status()  # Raise an error if Serper said no (for example, a wrong API key).
    return response.json()


def format_results(results: list[dict]) -> str:
    """Turn search results into plain text that the LLM can read."""
    text = f"Today is {date.today()}.\n\n"  # Helps the LLM judge how recent a result is.
    for result in results[:5]:
        text += f"{result['title']}\n{result['link']}\n{result.get('snippet', '')}\n\n"
    print('text :::::' , text)    
    return text


@tool
def web_search(query: str) -> str:
    """Search Google. Returns the top results, each with a title, link and short snippet.

    To read a result in full, call read_webpage with its link.
    """
    print(f"   [TOOL] web_search called with query: {query!r}")
    try:
        data = call_serper("https://google.serper.dev/search", {"q": query})
    except httpx.HTTPError as error:
        print(f"   [TOOL] web_search failed: {error}")
        return f"The search failed: {error}"  # The LLM reads this and tells the user.

    results = data.get("organic", [])
    print(f"   [TOOL] web_search got {len(results)} results. Top links:")
    for result in results[:3]:
        print(f"            {result['link']}")
    return format_results(results)


@tool
def news_search(query: str) -> str:
    """Search Google News. Returns recent articles, each with a title, link and short snippet.

    To read an article in full, call read_webpage with its link.
    """
    print(f"   [TOOL] news_search called with query: {query!r}")
    try:
        data = call_serper("https://google.serper.dev/news", {"q": query})
    except httpx.HTTPError as error:
        print(f"   [TOOL] news_search failed: {error}")
        return f"The search failed: {error}"

    results = data.get("news", [])
    print(f"   [TOOL] news_search got {len(results)} articles. Top links:")
    for result in results[:3]:
        print(f"            {result['link']}")
    return format_results(results)


@tool
def read_webpage(url: str) -> str:
    """Open a web page and return its text."""
    print('READ_WEBPAGE TOOL CALLING THIS URL ', url)
    try:
        data = call_serper("https://scrape.serper.dev", {"url": url})
    except httpx.HTTPError as error:
        print(f"   [TOOL] read_webpage failed: {error}")
        return f"Could not open the page: {error}"

    # Only send the start of the page. Long pages use up Groq's free limit
    # (8,000 tokens per minute) very quickly.
    text = data.get("text", "")[:3000]
    print(f"   [TOOL] read_webpage got {len(text)} characters of text")
    return text


@tool
def places_search(query: str) -> str:
    """Search Google Maps for places, like restaurants, shops, hotels or hospitals.

    Put the city or area in the query, for example "biryani restaurants in Lahore".
    Returns each place's name, category, address, rating, phone and website.
    """
    print(f"   [TOOL] places_search called with query: {query!r}")
    try:
        data = call_serper("https://google.serper.dev/places", {"q": query})
    except httpx.HTTPError as error:
        print(f"   [TOOL] places_search failed: {error}")
        return f"The search failed: {error}"

    places = data.get("places", [])
    print(f"   [TOOL] places_search got {len(places)} places. Top ones:")
    for place in places[:3]:
        print(f"            {place['title']} (rating {place.get('rating', '-')})")

    # Turn the top 5 places into plain text the LLM can read.
    # Some places have no phone or website, so .get() gives a default instead.
    text = ""
    for place in places[:5]:
        text += f"{place['title']} ({place.get('category', 'place')})\n"
        text += f"Address: {place.get('address', 'not listed')}\n"
        text += f"Rating: {place.get('rating', 'no rating')} from {place.get('ratingCount', 0)} reviews\n"
        text += f"Phone: {place.get('phoneNumber', 'not listed')}\n"
        text += f"Website: {place.get('website', 'not listed')}\n\n"
    return text or "No places found."


# ---------------------------------------------------------------------------
# Math tools (plain Python, no API key needed)
# ---------------------------------------------------------------------------


@tool
def add(a: float, b: float) -> float:
    """Add two numbers: a + b."""
    result = a + b
    print(f"   [TOOL] add({a}, {b}) = {result}")
    return result


@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers: a - b."""
    result = a - b
    print(f"   [TOOL] subtract({a}, {b}) = {result}")
    return result


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers: a * b."""
    result = a * b
    print(f"   [TOOL] multiply({a}, {b}) = {result}")
    return result


@tool
def divide(a: float, b: float) -> float | str:
    """Divide two numbers: a / b."""
    if b == 0:
        print(f"   [TOOL] divide({a}, {b}) = error, cannot divide by zero")
        return "Error: cannot divide by zero."
    result = a / b
    print(f"   [TOOL] divide({a}, {b}) = {result}")
    return result
