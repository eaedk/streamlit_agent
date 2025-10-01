# streamlit_app.py — LangChain 0.3 (fixed)

import os, re
from dotenv import load_dotenv
import streamlit as st
from streamlit.components.v1 import html as html_component

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults  
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Setup ---
load_dotenv()

st.set_page_config(page_title="LangChain 0.3: Chat with Search", page_icon="🦜")
st.title("🦜 LangChain 0.3: Chat with search")

openai_api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")
model_name = os.getenv("MODEL_NAME") or "gpt-4o-mini"

# Streamlit-persisted chat history
msgs = StreamlitChatMessageHistory()
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

def looks_like_html(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    return bool(re.search(r"<(div|p|table|ul|ol|li|span|section|article|h[1-6]|style|script|header|footer)[\s>]", t, re.I))

# Render past messages + any saved intermediate steps
avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    if msg.type not in avatars:
        continue
    with st.chat_message(avatars[msg.type]):
        for step in st.session_state.get("steps", {}).get(str(idx), []):
            if getattr(step[0], "tool", "") == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        # IMPORTANT: sandbox AI HTML to prevent CSS bleed
        if msg.type == "ai" and looks_like_html(msg.content):
            html_component(msg.content, height=600, scrolling=True,)
        else:
            st.write(msg.content)

# Build LLM + tools + agent once (outside chat submit)
if openai_api_key:
    llm = ChatOpenAI(model=model_name, api_key=openai_api_key, streaming=True)
    tools = [#DuckDuckGoSearchRun(name="Search"),
             DuckDuckGoSearchResults(name="Search")]

    # Prompt with chat history + tool scratchpad
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant that can use tools when needed. "
                    "Return a self-contained HTML block designed for rendering inside an iframe "
                    "(Streamlit components.html). "
                    "Scope ALL styles to a single top-level <div id='app'> without using global selectors "
                    "like html, body, *, :root. "
                    "No external JS or fonts; inline CSS only; keep it lightweight."
                ),
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=False,
    )

    # Wrap with LCEL message history (replaces ConversationBufferMemory)
    def get_history(_session_id: str):
        return msgs

    agent_with_history = RunnableWithMessageHistory(
        executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# Chat input
if prompt_text := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.chat_message("user").write(prompt_text)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        cfg["configurable"] = {"session_id": "default"}  # required by RunnableWithMessageHistory

        # Run the agent with chat history managed automatically
        response = agent_with_history.invoke({"input": prompt_text}, cfg)

        # Sandbox the latest HTML too (unique key avoids widget clashes)
        html_component(response["output"], height=600, scrolling=True,)

        # Save intermediate tool steps aligned to the latest AI message index
        ai_idx = len(msgs.messages) - 1
        st.session_state.steps[str(ai_idx)] = response.get("intermediate_steps", [])
