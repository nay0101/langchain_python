from langchain_community.agent_toolkits import SQLDatabaseToolkit
from helpers.config import Config
from sqlalchemy import create_engine
import pandas as pd
from helpers import get_llm
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import streamlit as st
import time
from streamlit import session_state as session
from langchain.globals import set_debug

set_debug(True)


def get_agent():
    llm = get_llm("gpt-4o-mini")
    engine = create_engine(f"{Config.POSTGRES_CONNECTION_STRING}/sql_agent")
    df = pd.read_excel("./sample_data/organizations_excel_small.xlsx", sheet_name=None)
    db = SQLDatabase(engine=engine)

    for sheet_name, df in df.items():
        table_name = sheet_name.replace(" ", "_").lower()
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        print(f"Inserted data from sheet '{sheet_name}' into table '{table_name}'")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()

    SQL_PREFIX = """You are an agent designed to interact with a SQL database.
      Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
      Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results unless user is specifically asking for more results.
      You can order the results by a relevant column to return the most interesting examples in the database.
      Never query for all the columns from a specific table, only ask for the relevant columns given the question.
      You have access to tools for interacting with the database.
      Only use the below tools. Only use the information returned by the below tools to construct your final answer.
      You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

      DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

      To start you should ALWAYS look at the tables in the database to see what you can query.
      Do NOT skip this step.
      Then you should query the schema of the most relevant tables."""

    system_message = SystemMessage(content=SQL_PREFIX)
    agent = create_react_agent(llm, tools, state_modifier=system_message)
    return agent


def response_generator(prompt):
    with st.spinner("Thinking..."):
        result = []
        for answer in session.agent.stream(
            {"messages": [HumanMessage(content=prompt)]}
        ):
            result.append(answer)
        response = result[-1]["agent"]["messages"][0].content
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


if "agent" not in session:
    session.agent = get_agent()

st.title("Agent")

if "messages_agent" not in session:
    session.messages_agent = []

for message in session.messages_agent:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Enter Your Message..."):
    session.messages_agent.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
        session.messages_agent.append({"role": "assistant", "content": response})
