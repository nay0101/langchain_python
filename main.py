import streamlit as st
from streamlit import session_state as session
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
    _SPARSE_MODEL_TYPES,
    _ELASTIC_HYBRID_SEARCH_TYPES,
)
from typing import get_args
import math

initial_config = {
    "chat_model": "gpt-4o",
    "embedding_model": "text-embedding-3-large",
    "ingest_embedding_model": "text-embedding-3-large",
    "dimension": 256,
    "ingest_dimension": 256,
    "vector_db": "elasticsearch",
    "ingest_vector_db": "elasticsearch",
    "index_name": "elastic_hybrid_search",
    "ingest_index_name": "elastic_hybrid_search",
    "hybrid_search": True,
    "ingest_hybrid_search": True,
    "hybrid_search_type": "dense_keyword",
    "ingest_hybrid_search_type": "dense_keyword",
    "sparse_model": ".elser_model_2",
    "ingest_sparse_model": ".elser_model_2",
    "weight": 0.5,
    "top_k": 5,
    "top_p": 0.9,
    "temperature": 0.1,
    "use_reranker": False,
    "reranker_model": "BAAI/bge-reranker-base",
    "score_threshold": 0.01,
    "instruction": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.",
}


for key, value in initial_config.items():
    if key not in session:
        session[key] = value


data_ingest_config = {
    "initial_url": "https://win066.wixsite.com/brillar-bank",
    "ignore_urls": "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-1,https://win066.wixsite.com/brillar-bank/brillar-bank-blog-2,https://win066.wixsite.com/brillar-bank/brillar-bank-blog-3,https://win066.wixsite.com/brillar-bank/brillar-bank-blog-4,",
    "crawling_method": "crawl_child_urls",
    "max_depth": 3,
    "chunk_size": 2000,
    "chunk_overlap": 200,
}

for key, value in data_ingest_config.items():
    if key not in session:
        session[key] = value


def dimension_available(embedding_model):
    return (
        embedding_model == "text-embedding-3-large"
        or embedding_model == "text-embedding-3-small"
    )


def add_prefix(prefix, key):
    return f"{prefix}_" + key if prefix else key


def is_hybrid_search(vector_db):
    return vector_db == "elasticsearch" or vector_db == "qdrant"


def initialize_components():

    if "llm" not in session:
        session.llm = get_llm(
            model_name=session.chat_model,
            temperature=session.temperature,
            top_p=session.top_p,
        )

    if "retriever" not in session:
        session.retriever = get_retriever(
            index_name=session.index_name,
            embedding_model=session.embedding_model,
            dimension=session.dimension,
            vector_db=session.vector_db,
            hybrid_search=session.hybrid_search,
            top_k=session.top_k,
            score_threshold=session.score_threshold,
        )

    if "reranker" not in session:
        session.reranker = get_reranker(
            base_retriever=session.retriever,
            model_name=session.reranker_model,
            top_k=session.top_k,
        )

    if "instruction" not in session:
        session.instruction = session.instruction


def get_chain():
    session.chain = create_conversational_retrieval_chain(
        llm=session.llm,
        retriever=(session.reranker if session.use_reranker else session.retriever),
        instruction=session.instruction,
    )


def save_response(path, question, answer, context):
    with open(path, "+a", encoding="utf-8") as file:
        file.write(f"Question: {question}\nAnswer: {answer}\n\n")
        # file.write(f"Context:\n\n")
        # for i in range(0, len(context)):
        #     file.write(f"Source: {context[i][1]}\n{context[i][0]}")
        #     file.write(
        #         "\n------------------------------------------------------------------------------------------------\n"
        #     )
        file.write(
            "------------------------------------------------------------------------------------------------\n"
        )


# Streamed response emulator
def response_generator(prompt):
    response = invoke_conversational_retrieval_chain(
        chain=session.chain,
        input=prompt,
        trace=False,
    )
    answer = response["answer"]
    source_documents = response["source_documents"]
    token_usage = response["token_usage"]
    # session.response
    session.token_usage = token_usage
    session.context = [[i["page_content"], i["source"]] for i in source_documents]
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)


initialize_components()

if "chain" not in session:
    get_chain()

st.title("Chat")
# Initialize chat history
if "messages" not in session:
    session.messages = []


# Display chat messages from history on app rerun
for message in session.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            response = message["content"]["response"]
            tokens = message["content"]["token_usage"]
            context = message["content"]["context"]
            st.markdown(response)
            left, right = st.columns(2, vertical_alignment="center")
            with left:
                st.text(
                    f'Input Tokens: {tokens["input_tokens"]}, Output Tokens: {tokens["output_tokens"]}'
                )
            # with right:
            #     save_response_button = st.button(
            #         label="Save Response",
            #         key=f"save_response_button_{session.messages.index(message)}",
            #     )

            # if save_response_button:
            #     with st.spinner("Saving..."):
            #         save_file_path = (
            #             f"./responses/{session.index_name}_{session.vector_db}.txt"
            #         )

            #         save_response(
            #             path=save_file_path,
            #             question=session.messages[session.messages.index(message) - 1][
            #                 "content"
            #             ],
            #             answer=response,
            #             context=context,
            #         )
            #     st.success("Response Saved.")

            st.text(f"Sources:")
            for i in range(0, len(context)):
                with st.popover(f"{context[i][1]}"):
                    st.markdown(context[i][0])
        else:
            st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("Enter you message..."):
    # Add user message to chat history
    session.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
        tokens = session.token_usage
        context = session.context
        left, right = st.columns(2, vertical_alignment="center")
        with left:
            st.text(
                f'Input Tokens: {tokens["input_tokens"]}, Output Tokens: {tokens["output_tokens"]}'
            )
        # with right:
        #     save_response_button = st.button(
        #         label="Save Response",
        #         key=f"save_response_button_{len(session.messages)}",
        #     )

        #     if save_response_button:
        #         with st.spinner("Saving..."):
        #             save_file_path = (
        #                 f"./responses/{session.index_name}_{session.vector_db}.txt"
        #             )

        #             save_response(
        #                 path=save_file_path,
        #                 question=prompt,
        #                 answer=response,
        #                 context=context,
        #             )
        #         st.success("Response Saved.")

        st.text(f"Sources:")
        for i in range(0, len(context)):
            with st.popover(f"{context[i][1]}"):
                st.markdown(context[i][0])

    # Add assistant response to chat history
    session.messages.append(
        {
            "role": "assistant",
            "content": {
                "response": response,
                "token_usage": tokens,
                "context": context,
            },
        }
    )


def common_ui_configs(prefix=""):
    embedding_model = st.selectbox(
        label="Embedding Model",
        options=sorted(get_args(_EMBEDDING_TYPES)),
        key=add_prefix(prefix, "embedding_model"),
    )

    if dimension_available(embedding_model):
        embedding_dimension = st.number_input(
            label="Embedding Dimension",
            key=add_prefix(prefix, "dimension"),
            min_value=0,
        )

    vector_db = st.selectbox(
        label="Vector DB",
        options=sorted(get_args(_VECTOR_DB)),
        key=add_prefix(prefix, "vector_db"),
    )

    index_name = st.text_input(
        label="Vector Index Name",
        key=add_prefix(prefix, "index_name"),
    )

    if is_hybrid_search(vector_db):
        hybrid_search = st.toggle(
            label="Hybrid Search",
            key=add_prefix(prefix, "hybrid_search"),
        )
        if hybrid_search:
            if vector_db == "elasticsearch":
                hybrid_search_type = st.selectbox(
                    label="Hybrid Search Type",
                    options=sorted(get_args(_ELASTIC_HYBRID_SEARCH_TYPES)),
                    key=add_prefix(prefix, "hybrid_search_type"),
                )
            if hybrid_search_type != "dense_keyword":
                sparse_model = st.selectbox(
                    label="Sparse Model",
                    options=sorted(get_args(_SPARSE_MODEL_TYPES)),
                    key=add_prefix(prefix, "sparse_model"),
                )


with st.sidebar:
    # Data Source
    with st.container(border=True):
        st.title("Add Data Source")
        link_to_crawl = st.text_input(
            label="Link to Crawl",
            key="initial_url",
        )

        crawl_options = sorted(get_args(_CRAWLING_TYPES))
        crawling_method = st.selectbox(
            label="Crawling Option",
            options=crawl_options,
            key="crawling_method",
        )

        ignore_urls = st.text_area(
            label="Ignore URLS",
            key="ignore_urls",
        )

        max_crawl_depth = st.number_input(
            label="Max Depth",
            min_value=0,
            max_value=5,
            key="max_depth",
        )

        chunk_size = st.slider(
            label="Chunk Size",
            min_value=100,
            max_value=5000,
            step=100,
            key="chunk_size",
        )

        chunk_overlap = st.slider(
            label="Chunk Overlap",
            min_value=0,
            max_value=int(math.floor(chunk_size / 2)),
            step=10,
            key="chunk_overlap",
        )

        common_ui_configs(prefix="ingest")

        ingest_data_button = st.button(label="Ingest Data", key="ingest_data_button")

        if ingest_data_button:
            with st.spinner("Crawling..."):
                start_time = time.time()
                urls = crawl(
                    start_url=session.initial_url,
                    crawling_method=session.crawling_method,
                    max_depth=session.max_depth,
                    ignore_list=session.ignore_urls.split(","),
                )
                elapsed_time = time.time() - start_time
            st.success(f"Crawling Finised. {elapsed_time}s")

            with st.spinner("Ingesting Data..."):
                start_time = time.time()
                ingest_data(
                    urls=urls,
                    embedding_model=session.ingest_embedding_model,
                    index_name=session.ingest_index_name,
                    dimension=session.ingest_dimension,
                    vector_db=session.ingest_vector_db,
                    chunk_size=session.chunk_size,
                    chunk_overlap=session.chunk_overlap,
                    hybrid_search=session.ingest_hybrid_search,
                    **{
                        "sparse_model": session.ingest_sparse_model,
                        "hybrid_search_type": session.ingest_hybrid_search_type,
                    },
                )

                session.embedding_model = session.ingest_embedding_model
                session.index_name = session.ingest_index_name
                session.dimension = session.ingest_dimension
                session.vector_db = session.ingest_vector_db
                session.hybrid_search = session.ingest_hybrid_search
                session.sparse_model = session.ingest_sparse_model
                session.hybrid_search_type = session.ingest_hybrid_search_type

                elapsed_time = time.time() - start_time
            st.success(f"Ingesting Finised. {elapsed_time}s")

    # Retriever Config
    with st.container(border=True):
        st.title("Retriever Config")

        common_ui_configs()

        top_k = st.slider(
            label="Top_K",
            min_value=1,
            max_value=20,
            key="top_k",
        )

        score_threshold = st.slider(
            label="Score Threshold",
            min_value=0.01,
            max_value=0.99,
            key="score_threshold",
        )

        use_reranker = st.toggle(label="Use Reranker", key="use_reranker")

        if use_reranker:
            reranker_model = st.selectbox(
                label="Reranker",
                options=sorted(get_args(_RERANKER_TYPES)),
                key="reranker_model",
            )

        update_retriever = st.button(label="Update Retriever", key="update_retriever")

        if update_retriever:
            with st.spinner("Updating Retriever..."):
                session.retriever = get_retriever(
                    index_name=session.index_name,
                    embedding_model=session.embedding_model,
                    dimension=session.dimension,
                    vector_db=session.vector_db,
                    hybrid_search=session.hybrid_search,
                    top_k=session.top_k,
                    score_threshold=session.score_threshold,
                    **{
                        "sparse_model": session.sparse_model,
                        "hybrid_search_type": session.hybrid_search_type,
                    },
                )

                if use_reranker:
                    session.reranker = get_reranker(
                        base_retriever=session.retriever,
                        model_name=session.reranker_model,
                        top_k=session.top_k,
                    )
                get_chain()
            st.success("Retriever Updated.")

    # LLM Config
    with st.container(border=True):
        st.title("LLM Config")

        chat_model = st.selectbox(
            label="Chat Model",
            options=sorted(get_args(_LLM_TYPES)),
            key="chat_model",
        )

        top_p = st.slider(
            label="Top_P",
            min_value=0.1,
            max_value=0.9,
            step=0.1,
            key="top_p",
        )

        temperature = st.slider(
            label="Temperature",
            min_value=0.1,
            max_value=0.9,
            step=0.1,
            key="temperature",
        )

        update_llm = st.button(label="Update LLM", key="update_llm")

        if update_llm:
            with st.spinner("Updating LLM..."):
                session.llm = get_llm(
                    model_name=session.chat_model,
                    temperature=session.temperature,
                    top_p=session.top_p,
                )
                get_chain()
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
                get_chain()
            st.success("Instruction Updated.")
