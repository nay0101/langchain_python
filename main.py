import streamlit as st
import time
from helpers import (
    invoke_conversational_retrieval_chain,
    create_conversational_retrieval_chain,
    get_llm,
    get_retriever,
    get_reranker,
)
import re


def initialize_chat():
    llm = get_llm(model_name="claude-3-haiku-20240307")
    retriever = get_retriever(
        index_name="qdrant_test",
        embedding_model="text-embedding-3-large",
        dimension=256,
        vector_db="qdrant",
        hybrid_search=False,
        top_k=5,
    )
    reranker = get_reranker(
        base_retriever=retriever, model_name="BAAI/bge-reranker-base", top_k=5
    )
    return create_conversational_retrieval_chain(llm=llm, retriever=reranker)


# Streamed response emulator
def response_generator(prompt):
    if "context" in st.session_state:
        del st.session_state["context"]
    response = invoke_conversational_retrieval_chain(
        chain=st.session_state.chain,
        input=prompt,
        trace=True,
        langfuse_args={"session_id": 123, "user_id": "user"},
    )
    answer = response["answer"]
    source_documents = response["source_documents"]
    token_usage = response["token_usage"]
    st.session_state.token_usage = token_usage
    st.session_state.context = [
        [i["page_content"], i["source"]] for i in source_documents
    ]
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)


if "chain" not in st.session_state:
    st.session_state.chain = initialize_chat()

st.title("Chat")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter you message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.title("Config")

    if "context" in st.session_state:
        st.text(st.session_state.token_usage)
        for i in range(0, len(st.session_state.context)):
            text = st.session_state.context[i][0]

            sentences = re.split(r"(?<=[.!?]) +", text)

            # Remove extra white spaces within sentences
            cleaned_sentences = [" ".join(sentence.split()) for sentence in sentences]

            # Join the sentences with a newline after each sentence
            cleaned_text = "\n\n".join(cleaned_sentences)
            st.text(
                f"Document {i}\nSource: {st.session_state.context[i][1]}\n{cleaned_text}\n---------------------"
            )
