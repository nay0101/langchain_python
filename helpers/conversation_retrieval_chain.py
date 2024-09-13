from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable
from typing import Any, Optional
from langfuse.callback import CallbackHandler
from .custom_types import _LangfuseArgs, _ChainResult
from .config import Config
from langchain_core.outputs import LLMResult
from langchain.callbacks.base import BaseCallbackHandler
import pandas as pd
from langchain_core.prompts import PromptTemplate
from .config import Config
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from .retriever import get_retriever
from langchain_core.runnables import Runnable, RunnablePassthrough, chain
from operator import itemgetter
from langchain.globals import set_debug

set_debug(True)


class LLMResultHandler(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if response.generations[0][0].message.usage_metadata:
            token_usage = response.generations[0][0].message.usage_metadata
        else:
            usage = response.generations[0][0].message.response_metadata["token_usage"]
            token_usage = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        self.response = token_usage


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

    system_prompt = (
        instruction + """\n\n{context}"""
        if instruction
        else """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
    \n\n
    {context}"""
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt, document_separator="\n")

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain


def invoke_conversational_retrieval_chain(
    chain: Runnable,
    input: str,
    trace: bool = True,
    langfuse_args: Optional[_LangfuseArgs] = None,
) -> _ChainResult:
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

    llm_result_handler = LLMResultHandler()
    result = chain.invoke(
        {"input": input, "chat_history": []},
        config={
            "callbacks": (
                [llm_result_handler, langfuse_handler]
                if langfuse_handler
                else [llm_result_handler]
            )
        },
    )

    answer = result["answer"]
    # source_documents = [
    #     {"page_content": doc.page_content, "source": doc.metadata["source"]}
    #     for doc in result["context"]
    # ]

    # token_usage = llm_result_handler.response

    output = {
        "answer": answer,
        # "source_documents": source_documents,
        # "token_usage": token_usage,
    }

    if langfuse_handler:
        langfuse_handler.flush()

    return output


def custom_chain(
    llm: LanguageModelLike, retriever: RetrieverLike, instruction: Optional[str] = None
) -> Runnable:
    engine = create_engine(Config.POSTGRES_CONNECTION_STRING)

    sql_instruction = """
        You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query, and then ONLY return the plain query. No markdown format or explanation is needed.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

        Only use the following tables:
        {table_info}

        Question: {input}
        """
    sql_prompt = PromptTemplate.from_template(template=sql_instruction)

    db = SQLDatabase(engine=engine)

    write_query = create_sql_query_chain(llm, db, prompt=sql_prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    sql_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question. You don't need to tell the user that you are using SQL.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    contextualize_instructions = """Convert the latest user question into a standalone question given the chat history. Don't answer the question, return the question and nothing else (no descriptive text)."""
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_instructions),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    contextualize_question = contextualize_prompt | llm | StrOutputParser()

    qa_instructions = (
        """Answer the user question given the following context:\n\n{context}."""
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_instructions), ("human", "{question}")]
    )

    retriever = get_retriever(
        index_name="test",
        embedding_model="text-embedding-3-large",
        dimension=256,
        vector_db="qdrant",
    )

    routing_retriever = get_retriever(
        index_name="test",
        embedding_model="text-embedding-3-large",
        dimension=256,
        vector_db="qdrant",
        top_k=1,
    )

    @chain
    def contextualize_if_needed(input_: dict) -> Runnable:
        if input_.get("chat_history"):
            return contextualize_question
        else:
            return RunnablePassthrough() | itemgetter("question")

    @chain
    def data_routing(input_: dict) -> Runnable:
        doc = routing_retriever.invoke(input_["question"])
        if doc[0].metadata.get("search_type"):
            return (
                RunnablePassthrough.assign(query=write_query).assign(
                    result=itemgetter("query") | execute_query
                )
                | sql_prompt
            )
        else:
            return {
                "question": itemgetter("question"),
                "context": itemgetter("question") | retriever,
            } | qa_prompt

    final_chain = (
        {"question": contextualize_if_needed} | data_routing | llm | StrOutputParser()
    )

    return final_chain


def custom_invoke(
    chain: Runnable,
    input: str,
    trace: bool = True,
    langfuse_args: Optional[_LangfuseArgs] = None,
) -> _ChainResult:
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

    llm_result_handler = LLMResultHandler()
    result = chain.invoke(
        {"question": input, "chat_history": []},
        config={
            "callbacks": (
                [llm_result_handler, langfuse_handler]
                if langfuse_handler
                else [llm_result_handler]
            )
        },
    )

    answer = result
    # answer = result["answer"]
    # source_documents = [
    #     {"page_content": doc.page_content, "source": doc.metadata["source"]}
    #     for doc in result["context"]
    # ]

    # token_usage = llm_result_handler.response

    output = {
        "answer": answer,
        # "source_documents": source_documents,
        # "token_usage": token_usage,
    }

    if langfuse_handler:
        langfuse_handler.flush()

    return output
