from langchain_elasticsearch import (
    ElasticsearchStore,
    DenseVectorStrategy,
    SparseVectorStrategy,
)
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, get_args, Optional, Any
from langchain_core.vectorstores.base import VectorStore
import chromadb
from .custom_types import (
    _VECTOR_DB,
    _EMBEDDING_TYPES,
    _SPARSE_MODEL_TYPES,
    _ELASTIC_HYBRID_SEARCH_TYPES,
)
from .embedding_models import get_embedding_model
from .config import Config


def get_vector_store_instance(
    embedding_model: _EMBEDDING_TYPES,
    index_name: str,
    dimension: Optional[int] = None,
    vector_db: _VECTOR_DB = "chromadb",
    hybrid_search: bool = False,
    **kwargs,
) -> VectorStore:
    embedding = get_embedding_model(embedding_model, dimension)

    db_types = get_args(_VECTOR_DB)
    options = dict(zip(db_types, db_types))

    if vector_db not in options or vector_db == options["chromadb"]:
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        vector_store = Chroma(
            collection_name=index_name,
            client=chroma_client,
            embedding_function=embedding,
        )
    elif vector_db == options["elasticsearch"]:
        hybrid_search_type: _ELASTIC_HYBRID_SEARCH_TYPES = kwargs.get(
            "hybrid_search_type", "dense_keyword"
        )

        if hybrid_search_type == "dense_keyword":
            vector_store = ElasticsearchStore(
                es_cloud_id=Config.ELASTIC_CLOUD_ID,
                es_api_key=Config.ELASTIC_API_KEY,
                embedding=embedding,
                index_name=index_name,
                strategy=DenseVectorStrategy(hybrid=hybrid_search),
            )
        elif hybrid_search == "sparse_keyword":
            vector_store = ElasticsearchStore(
                es_cloud_id=Config.ELASTIC_CLOUD_ID,
                es_api_key=Config.ELASTIC_API_KEY,
                embedding=embedding,
                index_name=index_name,
                strategy=SparseVectorStrategy(model_id=sparse_model),
            )
        elif hybrid_search_type == "dense_sparse":
            dense_vector_store = ElasticsearchStore(
                es_cloud_id=Config.ELASTIC_CLOUD_ID,
                es_api_key=Config.ELASTIC_API_KEY,
                embedding=embedding,
                index_name=index_name,
                strategy=DenseVectorStrategy(hybrid=False),
            )
            sparse_vector_store = ElasticsearchStore(
                es_cloud_id=Config.ELASTIC_CLOUD_ID,
                es_api_key=Config.ELASTIC_API_KEY,
                index_name=f"{index_name}_sparse",
                strategy=SparseVectorStrategy(model_id=".elser_model_2"),
            )
            vector_store = [dense_vector_store, sparse_vector_store]
    elif vector_db == options["qdrant"]:
        sparse_model: _SPARSE_MODEL_TYPES = kwargs.get("sparse_model", "Qdrant/bm25")
        vector_store = QdrantVectorStore.construct_instance(
            client_options={"api_key": Config.QDRANT_API_KEY, "url": Config.QDRANT_URL},
            embedding=embedding,
            collection_name=index_name,
            vector_name="vectors",
            sparse_embedding=(
                FastEmbedSparse(model_name=sparse_model) if hybrid_search else None
            ),
            retrieval_mode=(
                RetrievalMode.HYBRID if hybrid_search else RetrievalMode.DENSE
            ),
        )
    else:
        pass

    return vector_store


def ingest_data(
    urls: str | List[str],
    embedding_model: _EMBEDDING_TYPES,
    index_name: str,
    dimension: Optional[int] = None,
    vector_db: _VECTOR_DB = "chromadb",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    hybrid_search: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    loader = WebBaseLoader(web_path=urls, requests_per_second=3)
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
        hybrid_search=hybrid_search,
        **kwargs,
    )

    hybrid_search_type: _ELASTIC_HYBRID_SEARCH_TYPES = kwargs.get(
        "hybrid_search_type", "dense_keyword"
    )

    if hybrid_search_type == "dense_keyword":
        vector_store.add_documents(docs)
    elif hybrid_search_type == "dense_sparse":
        vector_store[0].add_documents(docs)
        vector_store[1].add_documents(
            docs,
            bulk_kwargs={"request_timeout": 60},
        )

    return {
        "index_name": index_name,
        "vector_db": vector_db,
        "embedding_model": embedding_model,
        "dimension": dimension,
    }
