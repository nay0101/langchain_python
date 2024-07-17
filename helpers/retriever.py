from .custom_types import _VECTOR_DB
from typing import Optional
from langchain_core.retrievers import RetrieverLike
from .vector_store import get_vector_store_instance


def get_retriever(
    index_name: str,
    embedding_model: str,
    vector_db: _VECTOR_DB = "chromadb",
    dimension: Optional[int] = None,
    top_k: int = 4,
    score_threshold: float = 0.01,
    hybrid_search: bool = False,
) -> RetrieverLike:
    vector_store = get_vector_store_instance(
        embedding_model=embedding_model,
        index_name=index_name,
        dimension=dimension,
        vector_db=vector_db,
        hybrid_search=hybrid_search,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": score_threshold},
    )

    return retriever


__all__ = ["get_retriever"]
