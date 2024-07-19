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
from typing import List, Dict, get_args, Optional, Any
from langchain_core.vectorstores.base import VectorStore
from .custom_types import _VECTOR_DB, _EMBEDDING_TYPES
from .embedding_models import get_embedding_model
import chromadb
from .config import Config


def get_vector_store_instance(
    embedding_model: _EMBEDDING_TYPES,
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
            es_cloud_id=Config.ELASTIC_CLOUD_ID,
            es_api_key=Config.ELASTIC_API_KEY,
            embedding=embedding,
            index_name=index_name,
            strategy=DenseVectorStrategy(hybrid=hybrid_search),
        )
    else:
        pass

    return vector_store


def ingest_data(
    urls: List[str],
    embedding_model: _EMBEDDING_TYPES,
    index_name: str,
    dimension: Optional[int] = None,
    vector_db: _VECTOR_DB = "chromadb",
    chunk_size: int = 2000,
    chunk_overlap: int = 20,
) -> Dict[str, Any]:
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

    return {
        "index_name": index_name,
        "vector_db": vector_db,
        "embedding_model": embedding_model,
        "dimension": dimension,
    }
