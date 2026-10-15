"""A chat page for the Agents API.

Start the API first (uv run fastapi dev app/main.py), then in a second terminal:
    uv run streamlit run frontend.py
"""

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


def stream_answer(question: str):
    """Send the question to the API and yield the answer piece by piece."""
    payload = {"message": question, "thread_id": st.session_state.thread_id}
    with httpx.stream("POST", f"{API_URL}/agents/{agent_name}/chat/stream", json=payload, timeout=120) as response:
        st.session_state.thread_id = response.headers["X-Thread-Id"]  # So the agent remembers follow-up questions.
        yield from response.iter_text()


if question := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer = st.write_stream(stream_answer(question))  # The typing effect.
    st.session_state.messages.append({"role": "assistant", "content": answer})
