import streamlit as st
from streamlit import session_state as session
import time
from pathlib import Path
from helpers import (
    invoke_conversational_retrieval_chain,
    create_conversational_retrieval_chain,
    get_llm,
    get_retriever,
    get_reranker,
    crawl,
    ingest_data,
    load_csv,
    load_excel,
    text_to_speech,
    speech_to_text,
)
from helpers.custom_types import (
    _LLM_TYPES,
    _EMBEDDING_TYPES,
    _VECTOR_DB,
    _CRAWLING_TYPES,
    _RERANKER_TYPES,
    _ELASTIC_SPARSE_MODEL_TYPES,
    _QDRANT_SPARSE_MODEL_TYPES,
    _HYBRID_SEARCH_TYPES,
)
from typing import get_args
import math
import uuid
from audio_recorder_streamlit import audio_recorder

initial_config = {
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-large",
    "ingest_embedding_model": "text-embedding-3-large",
    "dimension": 256,
    "ingest_dimension": 256,
    "vector_db": "qdrant",
    "ingest_vector_db": "qdrant",
    "index_name": "audio_rag_test1",
    "ingest_index_name": "audio_rag_test1",
    "hybrid_search": False,
    "ingest_hybrid_search": False,
    "hybrid_search_type": None,
    "ingest_hybrid_search_type": None,
    "sparse_model": None,
    "ingest_sparse_model": None,
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
    "data_type": "URL",
    "initial_url": "https://win066.wixsite.com/brillar-bank",
    "ignore_urls": "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-1,https://win066.wixsite.com/brillar-bank/brillar-bank-blog-2,https://win066.wixsite.com/brillar-bank/brillar-bank-blog-3,https://win066.wixsite.com/brillar-bank/brillar-bank-blog-4,",
    "crawling_method": "crawl_child_urls",
    "max_depth": 3,
    "chunking": True,
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

    # if "retriever" not in session:
    #     session.retriever = get_retriever(
    #         index_name=session.index_name,
    #         embedding_model=session.embedding_model,
    #         dimension=session.dimension,
    #         vector_db=session.vector_db,
    #         hybrid_search=session.hybrid_search,
    #         top_k=session.top_k,
    #         score_threshold=session.score_threshold,
    #     )

    # if "reranker" not in session:
    #     session.reranker = get_reranker(
    #         base_retriever=session.retriever,
    #         model_name=session.reranker_model,
    #         top_k=session.top_k,
    #     )

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
def response_generator(prompt, response_file_path):
    response = invoke_conversational_retrieval_chain(
        chain=session.chain,
        input=prompt,
        trace=True,
    )
    answer = response["answer"]
    text_to_speech(answer, response_file_path)
    # source_documents = response["source_documents"]
    # token_usage = response["token_usage"]
    # session.token_usage = token_usage
    # session.context = [[i["page_content"], i["source"]] for i in source_documents]
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)


initialize_components()

# if "chain" not in session:
#     get_chain()

st.title("Audio Test")
# Initialize chat history

audio_bytes = audio_recorder(text="")

if audio_bytes:
    audio_file_path = Path("./audio/recorded.mp3")
    with open(audio_file_path, "wb") as audio_file:
        audio_file.write(audio_bytes)

    prompt = speech_to_text(audio_file_path)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_file_path = Path("./audio/response.mp3")
        response = st.write_stream(response_generator(prompt, response_file_path))
        with open(response_file_path, "rb") as response_audio:
            st.audio(response_audio)


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
            match vector_db:
                case "elasticsearch":
                    hybrid_search_type = st.selectbox(
                        label="Hybrid Search Type",
                        options=sorted(get_args(_HYBRID_SEARCH_TYPES)),
                        key=add_prefix(prefix, "hybrid_search_type"),
                    )
                    if hybrid_search_type == "dense_sparse":
                        sparse_model = st.selectbox(
                            label="Sparse Model",
                            options=sorted(get_args(_ELASTIC_SPARSE_MODEL_TYPES)),
                            key=add_prefix(prefix, "sparse_model"),
                        )
                case "qdrant":
                    sparse_model = st.selectbox(
                        label="Sparse Model",
                        options=sorted(get_args(_QDRANT_SPARSE_MODEL_TYPES)),
                        key=add_prefix(prefix, "sparse_model"),
                    )
                case _:
                    pass


with st.sidebar:
    # Data Source
    st.title("CONFIG")
    with st.container(border=True):
        st.title("Add Data Source")
        data_type = st.selectbox(
            label="Data Type", options=["URL", "File"], key="data_type"
        )
        match session.data_type:
            case "URL":
                with st.container(border=True):
                    st.title("Crawl")
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

            case "File":
                with st.container(border=True):
                    st.title("File Upload")
                    file_upload = st.file_uploader(
                        label="Upload a File", type=["csv", "xlsx"], key="file_upload"
                    )
                    if file_upload:
                        session.file_extension = file_upload.name.split(".")[-1]
                        session.file_name = f"{uuid.uuid4()}.{session.file_extension}"

                        bytes_data = file_upload.getvalue()
                        with open(
                            f"./uploaded_files/{session.file_name}",
                            "wb",
                        ) as file:
                            file.write(bytes_data)
            case _:
                pass

        chunking = st.toggle(label="Chunking", key="chunking")

        if session.chunking:
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
            if session.data_type == "URL":
                with st.spinner("Crawling..."):
                    start_time = time.perf_counter()
                    docs = crawl(
                        start_url=session.initial_url,
                        crawling_method=session.crawling_method,
                        max_depth=session.max_depth,
                        ignore_list=session.ignore_urls.split(","),
                    )
                    elapsed_time = time.perf_counter() - start_time
                st.success(f"Crawling Finised. {elapsed_time}s")

            elif session.data_type == "File":
                with st.spinner("Extracting..."):
                    start_time = time.perf_counter()
                    match session.file_extension:
                        case "xlsx":
                            docs = load_excel(f"./uploaded_files/{session.file_name}")
                        case "csv":
                            docs = load_csv(f"./uploaded_files/{session.file_name}")
                        case _:
                            print("Invalid File Type.")
                            pass
                    elapsed_time = time.perf_counter() - start_time
                st.success(f"Extracting Finised. {elapsed_time}s")

            with st.spinner("Ingesting Data..."):
                start_time = time.perf_counter()
                ingest_data(
                    documents=docs,
                    embedding_model=session.ingest_embedding_model,
                    index_name=session.ingest_index_name,
                    dimension=session.ingest_dimension,
                    vector_db=session.ingest_vector_db,
                    chunk_size=session.chunk_size,
                    chunk_overlap=session.chunk_overlap,
                    hybrid_search=session.ingest_hybrid_search,
                    sparse_model=session.ingest_sparse_model,
                    hybrid_search_type=session.ingest_hybrid_search_type,
                )

                session.embedding_model = session.ingest_embedding_model
                session.index_name = session.ingest_index_name
                session.dimension = session.ingest_dimension
                session.vector_db = session.ingest_vector_db
                session.hybrid_search = session.ingest_hybrid_search
                session.sparse_model = session.ingest_sparse_model
                session.hybrid_search_type = session.ingest_hybrid_search_type

                elapsed_time = time.perf_counter() - start_time
            st.success(f"Ingestion Finished. {elapsed_time}s")
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
                    sparse_model=session.sparse_model,
                    hybrid_search_type=session.hybrid_search_type,
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
