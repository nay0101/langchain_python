from .model_mapping import _LLMS
from .custom_types import _VENDORS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelLike


def get_llm(model_name: str, temperature: float = 0.7) -> LanguageModelLike:
    llm_vendor = _LLMS[model_name]

    if llm_vendor == _VENDORS["openai"]:
        llm = ChatOpenAI(model=model_name, temperature=temperature)
    elif llm_vendor == _VENDORS["googlegenai"]:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    elif llm_vendor == _VENDORS["anthropic"]:
        llm = ChatAnthropic(model_name=model_name, temperature=temperature)
    elif llm_vendor == _VENDORS["huggingface"]:
        base_model = HuggingFaceEndpoint(repo_id=model_name, temperature=temperature)
        llm = ChatHuggingFace(llm=base_model)
    else:
        print("Invalid LLM.")
        pass

    return llm


__all__ = ["get_llm"]
