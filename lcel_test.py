from operator import itemgetter
from helpers import get_llm, get_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, chain
import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.globals import set_debug

set_debug(True)

llm = get_llm("claude-3-5-sonnet-20240620")

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
engine = create_engine(POSTGRES_CONNECTION_STRING)
df = pd.read_excel("./sample_data/organizations_excel_small.xlsx", sheet_name=None)
# Iterate over the dictionary and insert each DataFrame into PostgreSQL
for sheet_name, df in df.items():
    # Optionally, clean up the sheet name to use as a table name
    table_name = sheet_name.replace(" ", "_").lower()

    # Insert the DataFrame into a table in PostgreSQL
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)

    print(f"Inserted data from sheet '{sheet_name}' into table '{table_name}'")

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
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question. You don't need to explain about the underlying processes. Just return the final result.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
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


full_chain = (
    {"question": contextualize_if_needed} | data_routing | llm | StrOutputParser()
)
result = full_chain.invoke(
    {
        "question": "what is the difference in amount of employees between 1st and 2nd organizations in terms of employee?",
        "chat_history": [
            ("human", "do you know the interest rates for fixed deposit?"),
            ("ai", "Yes"),
        ],
    }
)

print(result)
