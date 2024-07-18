from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable
from typing import Optional, Dict
from langfuse.callback import CallbackHandler
from .custom_types import _LANGFUSE_ARGS
from .config import Config


def create_conversational_retrieval_chain(
    llm: LanguageModelLike, retriever: RetrieverLike, instruction: Optional[str] = None
) -> Runnable:
    condense_question_system_template = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
    \n\n
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain


def invoke_conversational_retrieval_chain(
    chain: Runnable,
    input: str,
    trace: bool = True,
    langfuse_args: Optional[_LANGFUSE_ARGS] = None,
) -> Dict:
    langfuse_handler = (
        CallbackHandler(
            public_key=Config.LANGFUSE_PUBLIC_KEY,
            secret_key=Config.LANGFUSE_SECRET_KEY,
            host=Config.LANGFUSE_BASEURL,
            **langfuse_args if langfuse_args else {},
        )
        if trace
        else None
    )
    result = chain.invoke(
        {"input": input, "chat_history": []},
        config={"callbacks": [langfuse_handler] if langfuse_handler else None},
    )

    answer = result["answer"]
    source_documents = [
        {"page_content": doc.page_content, "source": doc.metadata["source"]}
        for doc in result["context"]
    ]

    if langfuse_handler:
        langfuse_handler.flush()

    return {"answer": answer, "source_documents": source_documents}
