import streamlit as st

rag_page = st.Page(
    "page1_rag.py", title="RAG", icon=":material/smart_toy:", default=True
)
agent_page = st.Page("page2_agent.py", title="AGENT", icon=":material/robot:")


pg = st.navigation([rag_page, agent_page])
st.set_page_config(
    page_title="Chat mal lay kg layy", page_icon=":material/quick_phrases:"
)
pg.run()
