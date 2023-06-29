import logging

import streamlit as st
from streamlit_chat import message

from lugpt import QueryHandler

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the logging format
)

st.title("💬 Kanton Luzern GPT")

handler = QueryHandler(openai_api_key=st.secrets["OPENAI_API_KEY"],
                       milvus_api_key=st.secrets["MILVUS_API_KEY"])
logging.info(f"QueryHandler initialized: {handler}")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "Wie kann ich helfen?"}]

if "assistant_responses" not in st.session_state:
    st.session_state["assistant_responses"] = []

with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Ihre nachricht:",
        placeholder="Was möchten Sie wissen?",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

for idx, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=idx)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True)

    # Progress indicator
    with st.spinner("Bereite die Antwort vor..."):
        # Updating conversation history before making a new call
        handler.chat_history = [(msg['role'], msg['content']) for msg in st.session_state.messages]

        logging.info(f"Retrieving answer with history: {handler.chat_history}")
        result = handler.get_answer(user_input)  # Pass the list of past responses
        result_de, sources_de = handler.process_output(result)
        result_de_with_sources = " ".join([result_de, sources_de])
        logging.info(f"Output after formatting: {result_de_with_sources}")

    msg = {"role": "assistant", "content": result_de_with_sources}
    st.session_state.messages.append(msg)
    message(msg["content"])
