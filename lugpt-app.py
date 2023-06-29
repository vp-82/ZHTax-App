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
        # history = [(msg['role'], msg['content']) for msg in st.session_state.messages]
        # Initialize an empty list for the chat history
        history = []

        # We start by assuming the first message is from the user
        is_user = True

        # The user's query and assistant's reply
        user_query = None
        assistant_reply = None

        # Iterate through each message in the session state messages
        for msg in st.session_state.messages:
            # If the message is from the user, save it as the user's query
            if is_user:
                user_query = msg['content']
            else:
                # If the message is from the assistant, save it as the assistant's reply
                assistant_reply = msg['content']

                # Once we have both the user's query and the assistant's reply, add them as a tuple to the chat history
                history.append((user_query, assistant_reply))

                # Reset the user's query and the assistant's reply
                user_query = None
                assistant_reply = None

            # Toggle the is_user flag
            is_user = not is_user

        # The last message from the user is the current question
        query = user_input
        logging.info(f"Retrieving answer with question: {query} and history: {history}")
        result = handler.get_answer(query=query, history=history)  # Pass the list of past responses
        logging.info(f"Output before formatting: {result}")
        result_de, sources_de = handler.process_output(result)
        result_de_with_sources = " ".join([result_de, sources_de])
        logging.info(f"Output after formatting: {result_de_with_sources}")

    msg = {"role": "assistant", "content": result_de_with_sources}
    st.session_state.messages.append(msg)
    message(msg)
    # message(msg["content"])
