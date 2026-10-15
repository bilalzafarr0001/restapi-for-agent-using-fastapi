# Agents API

A REST API (built with **FastAPI**) for three AI agents (built with **LangChain** and **LangGraph**). The LLM runs on **Groq**.

| Agent    | What it does                          | Tools                                   |
| -------- | ------------------------------------- | --------------------------------------- |
| `search` | Answers questions using Google        | `web_search`, `read_webpage`            |
| `news`   | Summarizes the latest news on a topic | `news_search`, `read_webpage`           |
| `math`   | Solves math problems step by step     | `add`, `subtract`, `multiply`, `divide` |

## The code: 3 files

```
app/
├── main.py     1. The API: receives your message, runs an agent, returns the answer
├── agents.py   2. The 3 agents: each one is an LLM + tools + instructions
└── tools.py    3. The tools: Python functions the agents can call
```

Read them in this order. Each file only uses the one below it.

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
cp .env.example .env             # 2. Put your two API keys in .env
uv run fastapi dev app/main.py   # 3. Start the server
```

Then open **http://127.0.0.1:8000/docs** in your browser. You can try every endpoint there.

API keys (both have a free tier):
- Groq: https://console.groq.com/keys
- Serper: https://serper.dev/api-keys

## Endpoints

| Method | Path                        | What it does                        |
| ------ | --------------------------- | ----------------------------------- |
| GET    | `/agents`                   | Lists the agents                    |
| POST   | `/agents/{agent_name}/chat` | Sends a message, returns the answer |

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

- **Model**: `qwen/qwen3.8-27b`. Groq marks it as a *preview* model, so it may be removed one day. If that happens, change the model name in `agents.py`. We don't use the `openai/gpt-oss` models: in these search agents they often try to call tools that don't exist (like `find` or `open`), and Groq rejects those replies.
- **Free tier limit**: Groq's free tier allows 8,000 tokens per minute. If you send many questions quickly, answers get slower (the app waits for the limit to reset), or you get a `502` error that mentions a rate limit.
- **Memory**: conversations are kept in memory, so restarting the server forgets them.
- **Serper credits**: a search costs 1 credit, reading a page costs 2.
- **Errors**: if a search fails, the agent says so in its answer. If Groq fails (wrong key, rate limit), you get a `502` with the reason.
