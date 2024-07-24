import streamlit as st
import time
from helpers import (
    invoke_conversational_retrieval_chain,
    create_conversational_retrieval_chain,
    get_llm,
    get_retriever,
    get_reranker,
    crawl,
    ingest_data,
)
from helpers.custom_types import (
    _LLM_TYPES,
    _EMBEDDING_TYPES,
    _VECTOR_DB,
    _CRAWLING_TYPES,
    _RERANKER_TYPES,
)
from typing import get_args
import re
import math

initial_config = {
    "llm": "claude-3-haiku-20240307",
    "embedding": "text-embedding-3-large",
    "dimension": 256,
    "vector_db": "qdrant",
    "index_name": "qdrant_hybrid_test_1",
    "hybrid_search": True,
    "top_k": 5,
    "top_p": 0.9,
    "temperature": 0.1,
    "reranker": "BAAI/bge-reranker-base",
    "score_threshold": 0.01,
    "instruction": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.",
}

data_ingest_config = {
    "initial_url": "https://win066.wixsite.com/brillar-bank",
    "ignore_urls": [
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-1",
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-2",
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-3",
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-4",
    ],
    "crawling_method": "crawl_child_urls",
    "max_depth": 3,
    "chunk_size": 2000,
    "chunk_overlap": 200,
}


def initialize_chat():
    st.session_state.llm = get_llm(
        model_name=initial_config["llm"],
        temperature=initial_config["temperature"],
        top_p=initial_config["top_p"],
    )
    st.session_state.retriever = get_retriever(
        index_name=initial_config["index_name"],
        embedding_model=initial_config["embedding"],
        dimension=initial_config["dimension"],
        vector_db=initial_config["vector_db"],
        hybrid_search=initial_config["hybrid_search"],
        top_k=initial_config["top_k"],
        score_threshold=initial_config["score_threshold"],
    )
    st.session_state.reranker = get_reranker(
        base_retriever=st.session_state.retriever,
        model_name=initial_config["reranker"],
        top_k=initial_config["top_k"],
    )
    st.session_state.instruction = initial_config["instruction"]

    return create_conversational_retrieval_chain(
        llm=st.session_state.llm,
        retriever=st.session_state.reranker,
        instruction=st.session_state.instruction,
    )


# Streamed response emulator
def response_generator(prompt):
    response = invoke_conversational_retrieval_chain(
        chain=st.session_state.chain,
        input=prompt,
        trace=False,
    )
    answer = response["answer"]
    source_documents = response["source_documents"]
    token_usage = response["token_usage"]
    # st.session_state.response
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
        if message["role"] == "assistant":
            response = message["content"]["response"]
            tokens = message["content"]["token_usage"]
            context = message["content"]["context"]
            st.markdown(response)
            st.text(
                f'Input Tokens: {tokens["input_tokens"]}, Output Tokens: {tokens["output_tokens"]}'
            )

            st.text(f"Sources:")
            for i in range(0, len(context)):
                with st.popover(f"{context[i][1]}"):
                    st.markdown(context[i][0])
        else:
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
        tokens = st.session_state.token_usage
        context = st.session_state.context
        st.text(
            f'Input Tokens: {tokens["input_tokens"]}, Output Tokens: {tokens["output_tokens"]}'
        )

        st.text(f"Sources:")
        for i in range(0, len(context)):
            with st.popover(f"{context[i][1]}"):
                st.markdown(context[i][0])

    # Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": {
                "response": response,
                "token_usage": tokens,
                "context": context,
            },
        }
    )

with st.sidebar:
    # Data Source
    with st.container(border=True):
        st.title("Add Data Source")
        link_to_crawl = st.text_input(
            label="Link to Crawl",
            key="link_to_crawl",
            value=data_ingest_config["initial_url"],
        )

        crawl_options = sorted(get_args(_CRAWLING_TYPES))
        crawling_method = st.selectbox(
            label="Crawling Option",
            options=crawl_options,
            key="crawling_method",
            index=crawl_options.index(data_ingest_config["crawling_method"]),
        )

        ignore_urls = st.text_area(
            label="Ignore URLS",
            key="ignore_urls",
            value=",".join(data_ingest_config["ignore_urls"]),
        )

        max_crawl_depth = st.number_input(
            label="Max Depth",
            min_value=0,
            max_value=5,
            key="max_crawl_depth",
            value=data_ingest_config["max_depth"],
        )

        chunk_size = st.slider(
            label="Chunk Size",
            min_value=100,
            max_value=5000,
            step=100,
            key="chunk_size",
            value=data_ingest_config["chunk_size"],
        )

        chunk_overlap = st.slider(
            label="Chunk Overlap",
            min_value=0,
            max_value=int(math.floor(chunk_size / 2)),
            step=10,
            key="chunk_overlap",
            value=data_ingest_config["chunk_overlap"],
        )

        embedding_options = sorted(get_args(_EMBEDDING_TYPES))
        embedding_model = st.selectbox(
            label="Embedding Model",
            options=embedding_options,
            key="embedding_model",
            index=embedding_options.index(initial_config["embedding"]),
        )

        if (
            embedding_model == "text-embedding-3-large"
            or embedding_model == "text-embedding-3-small"
        ):
            embedding_dimension = st.number_input(
                label="Embedding Dimension",
                value=initial_config["dimension"],
                key="embedding_dimension",
            )
        else:
            embedding_dimension = None

        vector_db_options = sorted(get_args(_VECTOR_DB))
        vector_db = st.selectbox(
            label="Vector DB",
            options=vector_db_options,
            key="vector_db",
            index=vector_db_options.index(initial_config["vector_db"]),
        )

        index_name = st.text_input(
            label="Vector Index Name",
            key="index_name",
            value=initial_config["index_name"],
        )

        if vector_db == "elasticsearch" or vector_db == "qdrant":
            hybrid_search = st.toggle(
                label="Hybrid Search",
                key="hybrid_search",
                value=initial_config["hybrid_search"],
            )
        else:
            hybrid_search = False

        ingest_data_button = st.button(label="Ingest Data", key="ingest_data_button")

        if ingest_data_button:
            with st.spinner("Crawling..."):
                start_time = time.time()
                urls = crawl(
                    start_url=link_to_crawl,
                    crawling_method=crawling_method,
                    max_depth=max_crawl_depth,
                    ignore_list=ignore_urls.split(","),
                )
                elapsed_time = time.time() - start_time
            st.success(f"Crawling Finised. {elapsed_time}s")

            with st.spinner("Ingesting Data..."):
                start_time = time.time()
                ingest_data(
                    urls=urls,
                    embedding_model=embedding_model,
                    index_name=index_name,
                    dimension=embedding_dimension,
                    vector_db=vector_db,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    hybrid_search=hybrid_search,
                )
                elapsed_time = time.time() - start_time
            st.success(f"Ingesting Finised. {elapsed_time}s")

    # Retriever Config
    with st.container(border=True):
        st.title("Retriever Config")

        embedding_options = sorted(get_args(_EMBEDDING_TYPES))
        retriever_embedding_model = st.selectbox(
            label="Embedding Model",
            options=embedding_options,
            key="retriever_embedding_model",
            index=embedding_options.index(initial_config["embedding"]),
        )

        if (
            retriever_embedding_model == "text-embedding-3-large"
            or retriever_embedding_model == "text-embedding-3-small"
        ):
            retriever_embedding_dimension = st.number_input(
                label="Embedding Dimension",
                key="retriever_embedding_dimension",
                value=initial_config["dimension"],
            )
        else:
            retriever_embedding_dimension = None

        vector_db_options = sorted(get_args(_VECTOR_DB))
        retriever_vector_db = st.selectbox(
            label="Vector DB",
            options=vector_db_options,
            key="retriever_vector_db",
            index=vector_db_options.index(initial_config["vector_db"]),
        )

        retriever_index_name = st.text_input(
            label="Vector Index Name",
            key="retriever_index_name",
            value=initial_config["index_name"],
        )

        if retriever_vector_db == "elasticsearch" or retriever_vector_db == "qdrant":
            retriever_hybrid_search = st.toggle(
                label="Hybrid Search",
                key="retriever_hybrid_search",
                value=initial_config["hybrid_search"],
            )
        else:
            retriever_hybrid_search = False

        top_k = st.slider(
            label="Top_K",
            min_value=1,
            max_value=20,
            key="top_k",
            value=initial_config["top_k"],
        )

        score_threshold = st.slider(
            label="Score Threshold",
            min_value=0.01,
            max_value=0.99,
            key="score_threshold",
            value=initial_config["score_threshold"],
        )

        toggle_reranker = st.toggle(
            label="Use Reranker", key="toggle_reranker", value=True
        )

        if toggle_reranker:
            reranker_options = sorted(get_args(_RERANKER_TYPES))
            reranker_model = st.selectbox(
                label="Reranker",
                options=reranker_options,
                key="reranker_model",
                index=reranker_options.index(initial_config["reranker"]),
            )

        update_retriever = st.button(label="Update Retriever", key="update_retriever")
        if update_retriever:
            with st.spinner("Updating Retriever..."):
                st.session_state.retriever = get_retriever(
                    index_name=retriever_index_name,
                    embedding_model=retriever_embedding_model,
                    dimension=retriever_embedding_dimension,
                    vector_db=retriever_vector_db,
                    hybrid_search=retriever_hybrid_search,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                if toggle_reranker:
                    st.session_state.reranker = get_reranker(
                        base_retriever=st.session_state.retriever,
                        model_name=reranker_model,
                        top_k=top_k,
                    )
                st.session_state.chain = create_conversational_retrieval_chain(
                    llm=st.session_state.llm,
                    retriever=(
                        st.session_state.reranker
                        if toggle_reranker
                        else st.session_state.retriever
                    ),
                    instruction=st.session_state.instruction,
                )
            st.success("Retriever Updated.")

    # LLM Config
    with st.container(border=True):
        st.title("LLM Config")

        llm_options = sorted(get_args(_LLM_TYPES))
        chat_model = st.selectbox(
            label="Chat Model",
            options=llm_options,
            key="chat_model",
            index=llm_options.index(initial_config["llm"]),
        )

        top_p = st.slider(
            label="Top_P",
            min_value=0.1,
            max_value=0.9,
            step=0.1,
            key="top_p",
            value=initial_config["top_p"],
        )
        temperature = st.slider(
            label="Temperature",
            min_value=0.1,
            max_value=0.9,
            step=0.1,
            key="temperature",
            value=initial_config["temperature"],
        )

        update_llm = st.button(label="Update LLM", key="update_llm")

        if update_llm:
            with st.spinner("Updating LLM..."):
                st.session_state.llm = get_llm(
                    model_name=chat_model, temperature=temperature, top_p=top_p
                )
                st.session_state.chain = create_conversational_retrieval_chain(
                    llm=st.session_state.llm,
                    retriever=(
                        st.session_state.reranker
                        if toggle_reranker
                        else st.session_state.retriever
                    ),
                    instruction=st.session_state.instruction,
                )
            st.success("LLM Updated.")

    # Instruction Config
    with st.container(border=True):
        st.title("Instruction")
        instruction = st.text_area(
            label="Instruction",
            key="instruction",
            height=200,
            label_visibility="hidden",
        )

        update_instruction = st.button(
            label="Update Instruction", key="update_instruction"
        )

        if update_instruction:
            with st.spinner("Updating Instruction..."):
                st.session_state.chain = create_conversational_retrieval_chain(
                    llm=st.session_state.llm,
                    retriever=(
                        st.session_state.reranker
                        if toggle_reranker
                        else st.session_state.retriever
                    ),
                    instruction=st.session_state.instruction,
                )
            st.success("Instruction Updated.")
