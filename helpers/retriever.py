from .custom_types import _VECTOR_DB, _EMBEDDING_TYPES, _HYBRID_SEARCH_TYPES
from typing import Optional, Dict
from langchain_core.retrievers import RetrieverLike
from langchain.retrievers import EnsembleRetriever
from .vector_store import get_vector_store_instance
from .embedding_models import get_embedding_model
from langchain_elasticsearch import ElasticsearchRetriever
from .config import Config


def get_retriever(
    index_name: str,
    embedding_model: _EMBEDDING_TYPES,
    vector_db: _VECTOR_DB = "chromadb",
    dimension: Optional[int] = None,
    top_k: int = 4,
    score_threshold: float = 0.01,
    hybrid_search: bool = False,
    **kwargs
) -> RetrieverLike:
    vector_store = get_vector_store_instance(
        embedding_model=embedding_model,
        index_name=index_name,
        dimension=dimension,
        vector_db=vector_db,
        hybrid_search=hybrid_search,
        **kwargs,
    )

    if hybrid_search:
        embeddings = get_embedding_model(
            embedding_model=embedding_model, dimension=dimension
        )

        def hybrid_query(search_query: str) -> Dict:
            vector = embeddings.embed_query(
                search_query
            )  # same embeddings as for indexing
            return {
                "query": {
                    "sparse_vector": {
                        "field": "vector.tokens",
                        "inference_id": ".elser_model_2",
                        "query": search_query,
                    }
                },
                "size": 2,
            }
            return {
                "query": {
                    "match": {
                        "text": search_query,
                    },
                },
                "knn": {
                    "field": "vector",
                    "query_vector": vector,
                    "k": 5,
                    "num_candidates": 10,
                },
                "rank": {"rrf": {}},
            }

        hybrid_search_type: _HYBRID_SEARCH_TYPES = kwargs.get(
            "hybrid_search_type", None
        )

        if hybrid_search_type is None or hybrid_search_type == "dense_keyword":
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            # retriever = ElasticsearchRetriever.from_es_params(
            #     index_name=index_name,
            #     body_func=hybrid_query,
            #     content_field="text",
            #     api_key=Config.ELASTIC_API_KEY,
            #     cloud_id=Config.ELASTIC_CLOUD_ID,
            # )

        elif hybrid_search_type == "dense_sparse":
            dense_retriever = vector_store[0].as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            sparse_retriever = vector_store[1].as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            weight = kwargs.get("weight", 0.5)
            retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=[weight, 1 - weight],
            )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )

    return retriever
