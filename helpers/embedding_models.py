import os
from langchain_core.embeddings import Embeddings
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from .model_mapping import _EMBEDDING_MODELS
from .custom_types import _VENDORS
from .config import Config


def get_embedding_model(
    embedding_model: str, dimension: Optional[int] = None
) -> Optional[Embeddings]:
    embedding_vendor = _EMBEDDING_MODELS[embedding_model]

    if embedding_vendor == _VENDORS["openai"]:
        embedding = OpenAIEmbeddings(
            model=embedding_model, dimensions=dimension, api_key=Config.OPENAI_API_KEY
        )
    elif embedding_vendor == _VENDORS["googlegenai"]:
        embedding = GoogleGenerativeAIEmbeddings(
            model=embedding_model, google_api_key=Config.GOOGLE_API_KEY
        )
    elif embedding_vendor == _VENDORS["huggingface"]:
        embedding = HuggingFaceInferenceAPIEmbeddings(
            model_name=embedding_model, api_key=Config.HUGGINGFACEHUB_API_TOKEN
        )
    else:
        print("Invalid Embedding Model.")
        pass

    return embedding
