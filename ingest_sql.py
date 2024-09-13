from helpers import get_vector_store_instance
from langchain_core.documents import Document

vector_store = get_vector_store_instance(
    embedding_model="text-embedding-3-large",
    index_name="test",
    dimension=256,
    vector_db="qdrant",
)

vector_store.add_documents(
    [
        Document(
            page_content="This data is about the organizations and their data. This includes Organization ID, Name, Website, Country, Description, Founded Year, Industry, and Number of Employees.",
            metadata={"search_type": "xlsx"},
        )
    ]
)
