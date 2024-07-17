from typing import Literal

_VECTOR_DB = Literal["chromadb", "elasticsearch"]

_VENDORS = {
    "openai": "openai",
    "googlegenai": "googlegenai",
    "huggingface": "huggingface",
    "anthropic": "anthropic",
}

__all__ = ["_VECTOR_DB", "_VENDORS"]
