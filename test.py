from helpers import (
    create_conversational_retrieval_chain,
    invoke_conversational_retrieval_chain,
    get_retriever,
    get_llm,
    get_reranker,
)

retriever = get_retriever(
    index_name="newtestwithgoogle",
    embedding_model="text-embedding-004",
    vector_db="chromadb",
    hybrid_search=False,
)
reranker = get_reranker(base_retriever=retriever, model_name="BAAI/bge-reranker-base")

llm = get_llm(model_name="claude-3-5-sonnet-20240620", temperature=0.5)
chain = create_conversational_retrieval_chain(
    llm=llm, retriever=reranker, instruction=None
)
result = invoke_conversational_retrieval_chain(chain, "what is fixed deposit")
