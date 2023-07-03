import logging

import streamlit as st

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

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if st.button("Reset Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

if prompt := st.chat_input():

    message_dicts = [{'role': 'assistant',
                      'content': 'Wie kann ich helfen?'}]

    history = []
    user_message = None
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            user_message = msg['content']
        elif user_message is not None:
            history.append((user_message, msg['content']))
            user_message = None

    logging.info(f"Created hitory from message state: {history}")

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    logging.info(f"Calling handler.get_answer with question: {prompt} and hstory: {history}")
    response = handler.get_answer(query=prompt, history=history)
    answer, sources = handler.process_output(response)
    answer_with_sources = " ".join([answer, sources])
    logging.info(f"Adding answer to msg state: {answer}")
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer_with_sources)
