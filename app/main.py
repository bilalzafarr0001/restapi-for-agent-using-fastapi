"""The REST API.

Start it:  uv run fastapi dev app/main.py
Try it:    open http://127.0.0.1:8000/docs in your browser
"""

import sys
import uuid

import groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents import AGENTS

# Lets print() show any character (like ₹ or emojis) in every Windows terminal.
sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(title="Agents API")


class ChatRequest(BaseModel):
    message: str
    # Leave empty to start a new conversation. To continue one, send the
    # thread_id you got back from the previous answer.
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


@app.get("/agents")
def list_agents() -> list[str]:
    """List the names of all agents."""
    return list(AGENTS)


@app.post("/agents/{agent_name}/chat")
def chat(agent_name: str, request: ChatRequest) -> ChatResponse:
    """Send a message to an agent and get its answer."""
    # 1. Find the agent by its name.
    if agent_name not in AGENTS:
        raise HTTPException(status_code=404, detail=f"No agent called '{agent_name}'. Choose one of: {list(AGENTS)}")
    agent = AGENTS[agent_name]

    # 2. Pick the conversation. A new thread_id means a new conversation.
    thread_id = request.thread_id or str(uuid.uuid4())
 
    # 3. Run the agent. It may call tools several times before it answers.
    #    Each tool prints a [TOOL] line when the LLM calls it. If you see no
    #    [TOOL] lines, the LLM answered from its own memory without any tool.
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            {"configurable": {"thread_id": thread_id}},
        )
    except groq.APIError as error:
        # The call to Groq failed, for example because of a wrong API key or a rate limit.
        print(f"[API]  Groq error: {error}")
        raise HTTPException(status_code=502, detail=f"Groq error: {error}")

    # 4. The last message in the conversation is the agent's answer.
    answer = result["messages"][-1].content
    print("[API]  The LLM wrote the final answer:")
    print(answer)
    print("=============================================\n")
    return ChatResponse(answer=answer, thread_id=thread_id)
