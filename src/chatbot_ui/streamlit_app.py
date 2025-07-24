import streamlit as st
from qdrant_client import QdrantClient
from retrieval import rag_pipeline

from core.config import config 

import streamlit as st


qdrant_client = QdrantClient(
    url=config.QDRANT_URL
)



if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # output = run_llm(client, st.session_state.messages)
        output = rag_pipeline(prompt, qdrant_client)
        st.write(output["answer"].answer)
    st.session_state.messages.append({"role": "assistant", "content": output})