from .custom_types import _VENDORS

_EMBEDDING_MODELS = {
    "text-embedding-3-large": _VENDORS["openai"],
    "text-embedding-3-small": _VENDORS["openai"],
    "text-embedding-ada-002": _VENDORS["openai"],
    "text-embedding-004": _VENDORS["googlegenai"],
    "BAAI/bge-m3": _VENDORS["huggingface"],
}

_LLMS = {
    "gpt-3.5-turbo": _VENDORS["openai"],
    "gpt-4o": _VENDORS["openai"],
    "gpt-4-turbo": _VENDORS["openai"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1": _VENDORS["huggingface"],
    "claude-3-haiku-20240307": _VENDORS["anthropic"],
    "claude-3-5-sonnet-20240620": _VENDORS["anthropic"],
}

_RERANKERS = {
    "BAAI/bge-reranker-base": _VENDORS["huggingface"],
    "rerank-multilingual-v3.0": _VENDORS["cohere"],
}

__all__ = ["_EMBEDDING_MODELS", "_LLMS", "_RERANKERS"]
