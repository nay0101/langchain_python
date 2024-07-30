from .custom_types import _VECTOR_DB, _EMBEDDING_TYPES
from typing import Optional
from langchain_core.retrievers import RetrieverLike
from langchain.retrievers import EnsembleRetriever
from .vector_store import get_vector_store_instance


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

    hybrid_search_type = kwargs.get("hybrid_search_type", "default")

    if hybrid_search_type == "default":
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    elif hybrid_search_type == "sparse_hybrid":
        dense_retriever = vector_store[0].as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )
        sparse_retriever = vector_store[1].as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )
        weight = kwargs.get("weight", 0.5)
        retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever], weights=[weight, 1 - weight]
        )

    return retriever
