import os
from langchain_elasticsearch import (
    ElasticsearchStore,
)
from langchain_elasticsearch import (
    ElasticsearchStore,
    DenseVectorStrategy,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import List, Tuple, get_args, Optional
from langchain_core.vectorstores.base import VectorStore
from .custom_types import _VECTOR_DB
from .embedding_models import get_embedding_model
import chromadb

__ES_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
__ES_API_KEY = os.getenv("ELASTIC_API_KEY")

"""
def initialize_retriever(documents, index_name, embedding, type):
    if type == "bm25":
        bm25_vector_store = ElasticsearchStore.from_documents(
            documents=documents,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
            index_name=f"{index_name}_bm25",
            strategy=BM25Strategy(),
        )
        dense_vector_store = ElasticsearchStore.from_documents(
            documents=documents,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
            index_name=f"{index_name}_dense",
            embedding=embedding,
            strategy=DenseVectorStrategy(),
        )

        dense_retriever = dense_vector_store.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = bm25_vector_store.as_retriever(search_kwargs={"k": 5})

        retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever])
    elif type == "hybrid":
        vector_store = ElasticsearchStore.from_documents(
            documents=documents,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
            index_name=f"{index_name}_hybrid",
            embedding=embedding,
            strategy=DenseVectorStrategy(hybrid=True),
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    elif type == "sparse":
        dense_vector_store = ElasticsearchStore.from_documents(
            documents=documents,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
            index_name=f"{index_name}_dense",
            strategy=SparseVectorStrategy(model_id=".elser_model_2"),
        )
        sparse_vector_store = ElasticsearchStore.from_documents(
            documents=documents,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
            index_name=f"{index_name}_sparse",
            embedding=embedding,
            strategy=DenseVectorStrategy(),
        )
        dense_retriever = dense_vector_store.as_retriever(search_kwargs={"k": 5})
        sparse_retriever = sparse_vector_store.as_retriever(search_kwargs={"k": 5})

        retriever = EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever])
    else:
        vector_store = ElasticsearchStore.from_documents(
            documents=documents,
            es_cloud_id=es_cloud_id,
            es_api_key=es_api_key,
            index_name=f"{index_name}",
            embedding=embedding,
            strategy=DenseVectorStrategy(),
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    return retriever
"""


def get_vector_store_instance(
    embedding_model: str,
    index_name: str,
    dimension: Optional[int] = None,
    vector_db: _VECTOR_DB = "chromadb",
    hybrid_search: bool = False,
) -> VectorStore:
    embedding = get_embedding_model(embedding_model, dimension)

    db_types = get_args(_VECTOR_DB)
    options = dict(zip(db_types, db_types))

    if vector_db == options["chromadb"] or vector_db not in options:
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        vector_store = Chroma(
            collection_name=index_name,
            client=chroma_client,
            embedding_function=embedding,
        )
    elif vector_db == options["elasticsearch"]:
        vector_store = ElasticsearchStore(
            es_cloud_id=__ES_CLOUD_ID,
            es_api_key=__ES_API_KEY,
            embedding=embedding,
            index_name=index_name,
            strategy=DenseVectorStrategy(hybrid=hybrid_search),
        )
    else:
        pass

    return vector_store


def ingest_data(
    urls: List[str],
    embedding_model: str,
    index_name: str,
    dimension: Optional[int] = None,
    vector_db: _VECTOR_DB = "chromadb",
    chunk_size: int = 2000,
    chunk_overlap: int = 20,
) -> Tuple[str, str, str, Optional[int]]:
    loader = WebBaseLoader(urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(data)
    vector_store = get_vector_store_instance(
        embedding_model=embedding_model,
        index_name=index_name,
        dimension=dimension,
        vector_db=vector_db,
    )
    vector_store.add_documents(docs)

    return (index_name, vector_db, embedding_model, dimension)
