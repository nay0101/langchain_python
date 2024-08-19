from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI


def get_agent():
    df = pd.read_excel("./organizations_excel.xlsx")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=3,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent


if "agent" not in st.session_state:
    st.session_state.agent = get_agent()

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])
