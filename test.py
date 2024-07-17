from helpers import (
    create_conversational_retrieval_chain,
    invoke_conversational_retrieval_chain,
    get_retriever,
    get_llm,
    get_reranker,
)

retriever = get_retriever(
    index_name="demo_dense",
    embedding_model="text-embedding-3-large",
    vector_db="elasticsearch",
    dimension=256,
    hybrid_search=False,
)
reranker = get_reranker(base_retriever=retriever)

llm = get_llm(model_name="gpt-4o", temperature=0.5)
chain = create_conversational_retrieval_chain(
    llm=llm, retriever=reranker, instruction=None
)
result = invoke_conversational_retrieval_chain(chain, "what is fixed deposit")
print(result)
