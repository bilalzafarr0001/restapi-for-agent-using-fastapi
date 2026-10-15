import json
import sys
import uuid
import groq
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain.messages import AIMessage, ToolMessage
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


def event(**fields) -> str:
    """One message of the stream: a line of JSON."""
    return json.dumps(fields) + "\n"


@app.post("/agents/{agent_name}/chat/stream")
def chat_stream(agent_name: str, request: ChatRequest) -> StreamingResponse:
    """Like chat(), but streams what happens, live: each tool the agent runs, then the answer piece by piece."""
    if agent_name not in AGENTS:
        raise HTTPException(status_code=404, detail=f"No agent called '{agent_name}'. Choose one of: {list(AGENTS)}")
    agent = AGENTS[agent_name]

    thread_id = request.thread_id or str(uuid.uuid4())

    def events():
        try:
            # "custom" gives us what announce_tool (in agents.py) sends: a tool is starting.
            # "messages" gives us each finished tool, and each small piece of text the LLM writes.
            for mode, data in agent.stream(
                {"messages": [{"role": "user", "content": request.message}]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode=["custom", "messages"],
            ):
                if mode == "custom":
                    yield event(type="tool_start", **data)
                    continue

                message, metadata = data
                # Only the "tools" step really runs tools. A tool blocked by a limit gets its
                # ToolMessage from the limit middleware instead, and is not shown.
                if isinstance(message, ToolMessage) and metadata["langgraph_node"] == "tools":
                    yield event(type="tool_end", id=message.tool_call_id, status=message.status, output=str(message.content)[:1000])
                elif isinstance(message, AIMessage) and message.content:
                    yield event(type="text", content=message.content)
        except groq.APIError as error:
            yield event(type="text", content=f"\n\nGroq error: {error}")  # Too late for a 502: the answer has already started.

    # The body is now a stream of events, so the thread_id goes in a header.
    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"X-Thread-Id": thread_id})


@app.post("/chat")
def chat_with_assistant(request: ChatRequest) -> ChatResponse:
    """Send a message. No agent name needed: the assistant picks the right tools by itself."""
    return chat("assistant", request)
