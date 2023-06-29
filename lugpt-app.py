import streamlit as st
from streamlit_chat import message

from lugpt import QueryHandler

st.title("💬 Kanton Luzern GPT")

handler = QueryHandler(openai_api_key=st.secrets["OPENAI_API_KEY"],
                       milvus_api_key=st.secrets["MILVUS_API_KEY"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "How can I help you?"}]

if "assistant_responses" not in st.session_state:
    st.session_state["assistant_responses"] = []

with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
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
        result = handler.get_answer(user_input)  # Pass the list of past responses

    # Extract the answer from the result
    answer = result['result']
    answer_de, source_de = handler.process_output(answer)

    " ".join([answer_de, source_de])

    # # Store the answer in the list of past responses
    # st.session_state["assistant_responses"].append(answer)

    msg = {"role": "assistant", "content": answer}
    st.session_state.messages.append(msg)
    message(msg["content"])
