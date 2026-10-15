"""The agents.

An agent = an LLM + tools + instructions (the system prompt).

create_agent() builds the agent with LangGraph. When the agent runs, it loops:
  1. The LLM reads the conversation.
  2. If it needs a tool, it asks for one. The tool runs, and its result is added
     to the conversation. Then back to step 1.
  3. When it needs no more tools, the LLM writes the final answer.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from app.tools import (
    add,
    divide,
    multiply,
    news_search,
    places_search,
    read_webpage,
    subtract,
    web_search,
)

# Read GROQ_API_KEY and SERPER_API_KEY from the .env file.
load_dotenv()

# The LLM (the agent's "brain"), served by Groq. ChatGroq finds GROQ_API_KEY by itself.
# To try another model, change the name here. See https://console.groq.com/docs/models
llm = ChatGroq(model="qwen/qwen3.8-27b")


search_agent = create_agent(
    model=llm,
    tools=[web_search, read_webpage],
    system_prompt="""You are a research assistant.

If the user gives you a link, call read_webpage on that link and answer from what you read.

Otherwise, follow these steps:
1. Call web_search. Always do this, even if you think you know the answer.
2. Call read_webpage on the most useful link from the results, to get the full details.
3. Answer clearly and briefly, based on what you read.

End with a 'Sources' list of the links you used, written as plain URLs.""",
    # The checkpointer remembers each conversation (by its thread_id), so the user
    # can ask follow-up questions. It is kept in memory: a restart forgets everything.
    checkpointer=InMemorySaver(),
)

news_agent = create_agent(
    model=llm,
    tools=[news_search, read_webpage],
    system_prompt="""You are a news assistant.

- To find news, call news_search. Summarize the main stories as short bullet points.
  For each story, give the source, the date, and the link as a plain URL.
- If the user asks for details (for example "tell me more" or "full details"), or gives
  you a link, call read_webpage on that article's link and answer from what you read.""",
    checkpointer=InMemorySaver(),
)

math_agent = create_agent(
    model=llm,
    tools=[add, subtract, multiply, divide],
    system_prompt=(
        "You are a math tutor. Use the math tools for every calculation. When you are "
        "done, write the whole solution step by step, then the final answer."
    ),
    checkpointer=InMemorySaver(),
)

places_agent = create_agent(
    model=llm,
    tools=[places_search],
    system_prompt="""You are a local guide that finds places on Google Maps.

If the user didn't say which city or area, ask them first.
Otherwise, always call places_search, even if you think you know the places.

List the places as short bullet points. For each place, give its name, address,
rating and number of reviews, and its phone number or website if it has one.""",
    checkpointer=InMemorySaver(),
)


# The API finds agents in this dictionary by name.
AGENTS = {
    "search": search_agent,
    "news": news_agent,
    "math": math_agent,
    "places": places_agent,
}
