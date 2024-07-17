import streamlit as st
import time
from helpers.conversation_retrieval_chain import (
    invoke_conversational_retrieval_chain,
    create_conversational_retrieval_chain,
)
from dotenv import load_dotenv
import re
from helpers.llms import get_llm

from helpers.retriever import get_retriever
from helpers.conversation_retrieval_chain import (
    create_conversational_retrieval_chain,
    invoke_conversational_retrieval_chain,
)


load_dotenv()


def initialize_chat():
    llm = get_llm("gpt-4o")
    retriever = get_retriever(
        index_name="newfuck",
        embedding_model="BAAI/bge-m3",
        vector_db="chromadb",
        dimension=256,
    )
    return create_conversational_retrieval_chain(llm, retriever)


# Streamed response emulator
def response_generator(prompt):
    if "context" in st.session_state:
        del st.session_state["context"]
    response = invoke_conversational_retrieval_chain(st.session_state.chain, prompt)
    st.session_state.context = [
        [i.page_content, i.metadata["source"]] for i in response["context"]
    ]
    for word in response["answer"].split():
        yield word + " "
        time.sleep(0.05)


st.title("Simple chat")

if "chain" not in st.session_state:
    st.session_state.chain = initialize_chat()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
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
    if "context" in st.session_state:
        for i in range(0, len(st.session_state.context)):
            text = st.session_state.context[i][0]
            # lines = texts.split("\n")
            # cleaned_lines = [" ".join(line.split()) for line in lines]
            # cleaned_text = "\n".join(cleaned_lines)
            sentences = re.split(r"(?<=[.!?]) +", text)

            # Remove extra white spaces within sentences
            cleaned_sentences = [" ".join(sentence.split()) for sentence in sentences]

            # Join the sentences with a newline after each sentence
            cleaned_text = "\n\n".join(cleaned_sentences)
            st.text(
                f"Document {i}\nSource: {st.session_state.context[i][1]}\n{cleaned_text}\n---------------------"
            )
