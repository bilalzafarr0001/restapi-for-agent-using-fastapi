import sys
import uuid
import groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agents import AGENTS


sys.stdout.reconfigure(encoding="utf-8")
app = FastAPI(title="Agents API")


class ChatRequest(BaseModel):
    message: str
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
    if agent_name not in AGENTS:
        raise HTTPException(status_code=404, detail=f"No agent called '{agent_name}'. Choose one of: {list(AGENTS)}")
    agent = AGENTS[agent_name]

    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            {"configurable": {"thread_id": thread_id}},
        )
    except groq.APIError as error:
        print(f"[API]  Groq error: {error}")
        raise HTTPException(status_code=502, detail=f"Groq error: {error}")

    answer = result["messages"][-1].content
    print("************************************   [API]  The LLM wrote the final answer: ***********************************************")
    print(answer)
    
    return ChatResponse(answer=answer, thread_id=thread_id)


@app.post("/chat")
def chat_with_assistant(request: ChatRequest) -> ChatResponse:
    """Send a message. No agent name needed: the assistant picks the right tools by itself."""
    return chat("assistant", request)
