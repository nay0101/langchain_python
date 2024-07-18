from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import RetrieverLike


def get_reranker(
    base_retriever: RetrieverLike,
    model_name: str = "BAAI/bge-reranker-base",
    top_k: int = 3,
) -> ContextualCompressionRetriever:
    model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=model, top_n=top_k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever
