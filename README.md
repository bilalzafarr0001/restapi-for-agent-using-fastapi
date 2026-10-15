# Agents API

A REST API (built with **FastAPI**) for six AI agents (built with **LangChain** and **LangGraph**). The LLM runs on **Groq**.

| Agent       | What it does                                         | Tools                                   |
| ----------- | ---------------------------------------------------- | --------------------------------------- |
| `search`    | Answers questions using Google                       | `web_search`, `read_webpage`            |
| `news`      | Summarizes the latest news on a topic                | `news_search`, `read_webpage`           |
| `math`      | Solves math problems step by step                    | `add`, `subtract`, `multiply`, `divide` |
| `places`    | Finds places on Google Maps                          | `places_search`                         |
| `weather`   | Current temperature and weather condition in a city  | `get_weather`                           |
| `assistant` | Answers anything: picks from **all** the tools above | all of them                             |

## The code: 3 files

```
app/
├── main.py     1. The API: receives your message, runs an agent, returns the answer
├── agents.py   2. The 6 agents: each one is an LLM + tools + instructions
└── tools.py    3. The tools: Python functions the agents can call
```

Read them in this order. Each file only uses the one below it.

`frontend.py` is a separate chat page (built with **Streamlit**). It talks to the API over HTTP, like any other client.

## How one request flows

Say you send `POST /agents/search/chat` with `{"message": "What is the latest version of FastAPI?"}`:

```
 main.py      1. chat() receives the message and finds the "search" agent.
    |
    v
 agents.py    2. The agent sends your message to the LLM (on Groq).
    |         3. The LLM decides: "I need to search the web."
    |            It asks to call web_search("latest FastAPI version").
    v
 tools.py     4. web_search() asks Serper (Google) and gets back titles, links and snippets.
    |
    v
 agents.py    5. The results go back to the LLM. It may call another tool
    |            (for example read_webpage, to read a page), or it is ready to answer.
    |         6. The LLM writes the final answer.
    v
 main.py      7. chat() returns {"answer": "...", "thread_id": "..."}
```

Steps 2 to 5 are the **agent loop**. You don't write it yourself: `create_agent()` (in `agents.py`) builds it for you with LangGraph.

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs Python and the packages for you.

```bash
uv sync                          # 1. Install the packages
cp .env.example .env             # 2. Put your three API keys in .env
uv run fastapi dev app/main.py   # 3. Start the server
```

Then open **http://127.0.0.1:8000/docs** in your browser. You can try every endpoint there.

**Chat page:** keep the server running, and in a second terminal:

```bash
uv run streamlit run frontend.py
```

It opens **http://localhost:8501**: pick an agent, type a message, and the answer appears word by word.

API keys (all have a free tier):
- Groq: https://console.groq.com/keys
- Serper: https://serper.dev/api-keys
- WeatherAPI: https://www.weatherapi.com/my/

## Endpoints

| Method | Path                        | What it does                        |
| ------ | --------------------------- | ----------------------------------- |
| POST   | `/chat`                     | **Easiest:** send any message, the assistant picks the tools |
| GET    | `/agents`                   | Lists the agents                    |
| POST   | `/agents/{agent_name}/chat` | Sends a message to one specific agent |
| POST   | `/agents/{agent_name}/chat/stream` | Same, but streams the answer as plain text while the LLM writes it. The `thread_id` is in the `X-Thread-Id` header. |

Request:

```json
{ "message": "What is the latest version of FastAPI?" }
```

Response:

```json
{ "answer": "The latest version of FastAPI is ...", "thread_id": "7025bed6-..." }
```

**Follow-up questions:** send the `thread_id` back with your next message, and the agent remembers the conversation:

```json
{ "message": "When was it released?", "thread_id": "7025bed6-..." }
```

## Try it from PowerShell

```powershell
$body = @{ message = "What is the latest version of FastAPI?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/agents/search/chat -ContentType "application/json" -Body $body
```

## Add your own agent

1. In `tools.py`, write a function with `@tool` and a clear docstring.
2. In `agents.py`, create the agent with `create_agent(...)`, like the others.
3. Add it to the `AGENTS` dictionary at the bottom of `agents.py`.

That's it: `/agents/your_agent/chat` works right away.

## Run the tests

```bash
uv run pytest
```

The tests use fake API keys and never call Groq or Serper.

## Good to know

- **Model**: `openai/gpt-oss-20b`. To change it, edit one line in `agents.py`. `openai/gpt-oss-120b` is smarter, but each model has its own free daily limit (200,000 tokens per day). Sometimes gpt-oss calls a tool the wrong way and Groq rejects the reply; the agent then asks again (up to 2 more times).
- **Limits per question**: an agent reads at most 2 web pages and calls at most 6 tools for one question. This keeps answers fast and saves your free limits.
- **Free tier limit**: Groq's free tier allows 8,000 tokens per minute. If you send many questions quickly, answers get slower (the app waits for the limit to reset), or you get a `502` error that mentions a rate limit.
- **Memory**: conversations are kept in memory, so restarting the server forgets them.
- **Serper credits**: a search costs 1 credit, reading a page costs 2.
- **Weather**: the current temperature and condition, from [WeatherAPI.com](https://www.weatherapi.com). Needs `WEATHER_API_KEY` in `.env`.
- **Errors**: if a search fails, the agent says so in its answer. If Groq fails (wrong key, rate limit), you get a `502` with the reason.
