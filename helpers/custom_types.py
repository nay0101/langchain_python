from typing import Literal, TypedDict, Optional, Dict

_VECTOR_DB = Literal["chromadb", "elasticsearch"]

_VENDORS = {
    "openai": "openai",
    "googlegenai": "googlegenai",
    "huggingface": "huggingface",
    "anthropic": "anthropic",
    "cohere": "cohere",
}

_LANGFUSE_ARGS = TypedDict(
    "LANGFUSE_ARGS",
    {
        "session_id": Optional[str],
        "user_id": Optional[str],
        "trace_id": Optional[str],
        "metadata": Optional[Dict],
    },
)


__all__ = ["_VECTOR_DB", "_VENDORS", "_LANGFUSE_ARGS"]
