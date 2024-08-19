from helpers import ingest_data, crawl

urls = crawl(
    start_url="https://win066.wixsite.com/brillar-bank/",
    ignore_list=[
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-1",
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-2",
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-3",
        "https://win066.wixsite.com/brillar-bank/brillar-bank-blog-4",
    ],
)
result = ingest_data(
    urls=urls,
    embedding_model="text-embedding-3-large",
    index_name="new_chroma",
    vector_db="chromadb",
    dimension=256,
    hybrid_search=True,
)
print(result)
