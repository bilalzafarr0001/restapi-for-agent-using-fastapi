import groq
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware, wrap_tool_call
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

from app.tools import (
    add,
    divide,
    get_weather,
    multiply,
    news_search,
    places_search,
    read_webpage,
    subtract,
    web_search,
)

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")

llm_retry = ModelRetryMiddleware(max_retries=2, retry_on=(groq.BadRequestError,), on_failure="error")


def tool_failed(error: Exception) -> str:
    return f"ERROR: the tool failed ({error}). Tell the user it failed. Do not answer from memory."


tool_retry = ToolRetryMiddleware(max_retries=1, on_failure=tool_failed)

page_limit = ToolCallLimitMiddleware(tool_name="read_webpage", run_limit=2)
tool_limit = ToolCallLimitMiddleware(run_limit=6)


@wrap_tool_call
def announce_tool(request, handler):
    """Tell the chat page which tool is starting, right before its body runs.

    Tools blocked by page_limit or tool_limit never get here, so the page only shows tools that really ran.
    """
    tool_call = request.tool_call
    get_stream_writer()({"id": tool_call["id"], "name": tool_call["name"], "input": tool_call["args"]})
    return handler(request)


def make_agent(tools: list, system_prompt: str):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        # announce_tool comes before tool_retry, so a retried tool is announced only once.
        middleware=[announce_tool, llm_retry, tool_retry, page_limit, tool_limit],
        checkpointer=InMemorySaver(),
    )


search_agent = make_agent(
    tools=[web_search, read_webpage],
    system_prompt="""You are a research assistant.

If the user gives you a link, call read_webpage on that link and answer from what you read.

Otherwise, follow these steps:
1. Call web_search. Always do this, even if you think you know the answer.
2. Call read_webpage on the most useful link from the results, to get the full details.
3. Answer clearly and briefly, based on what you read.

End with a 'Sources' list of the links you used, written as plain URLs.
Don't use citation marks like 【1】.""",
)

news_agent = make_agent(
    tools=[news_search, read_webpage],
    system_prompt="""You are a news assistant.

- To find news, call news_search. Summarize the main stories as short bullet points.
  For each story, give the source, the date, and the link as a plain URL.
- Only call read_webpage if the user asks for details (for example "tell me more" or
  "full details") or gives you a link. Then read at most 2 pages.

Don't use citation marks like 【1】.""",
)

math_agent = make_agent(
    tools=[add, subtract, multiply, divide],
    system_prompt="""You are a math tutor. Use the math tools for every calculation.
When you are done, write the whole solution step by step, then the final answer.""",
)

places_agent = make_agent(
    tools=[places_search],
    system_prompt="""You are a local guide that finds places on Google Maps.

If the user didn't say which city or area, ask them first.
Otherwise, always call places_search, even if you think you know the places.

List the places as short bullet points. For each place, give its name, address,
rating and number of reviews, and its phone number or website if it has one.""",
)

weather_agent = make_agent(
    tools=[get_weather],
    system_prompt="""You are a weather assistant.

If the user didn't say which city, ask first.
Otherwise, always call get_weather for that city, and answer in one short sentence:
the city, the current temperature and the current weather condition.

You only know the weather right now. If the user asks about another day, say so.""",
)

assistant_agent = make_agent(
    tools=[web_search, news_search, read_webpage, places_search, get_weather, add, subtract, multiply, divide],
    system_prompt="""You are a helpful assistant. Pick the tools that fit the question:
- web_search: facts and up-to-date information from Google.
- news_search: recent news on a topic.
- read_webpage: read a page in full, from a search result link or a link the user gives you.
- places_search: places on Google Maps, like restaurants or hospitals. If the user didn't
  say which city or area, ask first.
- get_weather: the current temperature and weather condition in a city.
- add, subtract, multiply, divide: use these for every calculation.

You can use several tools for one question. For facts that may have changed (prices,
records, versions, who holds a job, anything "latest"), always search first, even if
you think you know the answer. If no tool is needed (like "hi" or "thank you"), just answer.

Answer clearly and briefly. If you used web_search, news_search or read_webpage, end
with a 'Sources' list of the links you used, written as plain URLs. Never invent a link.
Don't use citation marks like 【1】.""",
)


AGENTS = {
    "search": search_agent,
    "news": news_agent,
    "math": math_agent,
    "places": places_agent,
    "weather": weather_agent,
    "assistant": assistant_agent,
}
