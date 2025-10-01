# streamlit_app.py — LangChain 0.3

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
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
model_name = os.getenv("MODEL_NAME") or "gpt-4o"

# Streamlit-persisted chat history
msgs = StreamlitChatMessageHistory()
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

# Render past messages + any saved intermediate steps
avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    # Ignore system/tool messages if any show up
    if msg.type not in avatars:
        continue
    with st.chat_message(avatars[msg.type]):
        for step in st.session_state.get("steps", {}).get(str(idx), []):
            if getattr(step[0], "tool", "") == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

# Build LLM + tools + agent once (outside chat submit)
if openai_api_key:
    llm = ChatOpenAI(model=model_name, api_key=openai_api_key, streaming=True)
    tools = [DuckDuckGoSearchRun(name="Search")]

    # Prompt with chat history + tool scratchpad
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that can use tools when needed. Your rusult will be display in a streamlit app, adapt the format and the style for an interesting view using st.html ."),
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

        # Write final answer
        # st.write(response["output"])
        st.html(response["output"])
        

        # Save intermediate tool steps aligned to the latest AI message index
        ai_idx = len(msgs.messages) - 1
        st.session_state.steps[str(ai_idx)] = response.get("intermediate_steps", [])
