import streamlit as st

rag_page = st.Page(
    "page1_rag.py", title="NORMAL RAG", icon=":material/smart_toy:", default=True
)
audio_page = st.Page(
    "page4_rag_audio.py", title="AUDIO TEST", icon=":material/smart_toy:"
)
custom_rage_page = st.Page(
    "page2_custom_rag.py", title="CUSTOM RAG", icon=":material/smart_toy:"
)
agent_page = st.Page(
    "page3_sql_agent.py", title="SQL AGENT", icon=":material/smart_toy:"
)

pg = st.navigation([rag_page, audio_page, custom_rage_page, agent_page])
st.set_page_config(
    page_title="HLB Chat",
    page_icon=":material/quick_phrases:",
    initial_sidebar_state="collapsed",
)
pg.run()
