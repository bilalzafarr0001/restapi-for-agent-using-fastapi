"""A chat page for the Agents API.

Start the API first (uv run fastapi dev app/main.py), then in a second terminal:
    uv run streamlit run frontend.py
"""

import json

import httpx
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Agents Chat", page_icon="🤖")
st.title("🤖 Agents Chat")

# Streamlit runs this whole file again on every click, so we keep the chat in st.session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = None


def new_chat():
    st.session_state.messages = []
    st.session_state.thread_id = None


with st.sidebar:
    try:
        agents = httpx.get(f"{API_URL}/agents").json()
    except httpx.ConnectError:
        st.error("Can't reach the API. Start it first: uv run fastapi dev app/main.py")
        st.stop()

    # Each agent has its own memory, so switching agents starts a new chat.
    agent_name = st.selectbox("Agent", agents, index=agents.index("assistant"), on_change=new_chat)
    st.button("New chat", on_click=new_chat)

# Show the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

ICONS = {"running": "⏳", "success": "✅", "error": "❌"}


def tools_line(tools: dict) -> str:
    """For example: "🔧 news_search ✅ · read_webpage ⏳"."""
    return "🔧 " + " · ".join(f"{tool['name']} {ICONS[tool['status']]}" for tool in tools.values())


def stream_events(question: str):
    """Send the question to the API and yield what happens, one event at a time."""
    payload = {"message": question, "thread_id": st.session_state.thread_id}
    with httpx.stream("POST", f"{API_URL}/agents/{agent_name}/chat/stream", json=payload, timeout=120) as response:
        st.session_state.thread_id = response.headers["X-Thread-Id"]  # So the agent remembers follow-up questions.
        for line in response.iter_lines():
            if line:
                yield json.loads(line)


if question := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        tools_area = st.empty()  # The tools the agent uses. Only shown while the answer is being generated.
        answer_area = st.empty()
        tools, answer = {}, ""

        for event in stream_events(question):
            if event["type"] == "tool_start":
                tools[event["id"]] = {"name": event["name"], "status": "running"}
                tools_area.caption(tools_line(tools))
            elif event["type"] == "tool_end":
                tools[event["id"]]["status"] = event["status"]  # ✅ only once the tool has really finished.
                tools_area.caption(tools_line(tools))
            elif event["type"] == "text":
                answer += event["content"]
                answer_area.markdown(answer + "▌")  # The typing effect.

        tools_area.empty()  # The answer is ready: hide the tools.
        answer_area.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
