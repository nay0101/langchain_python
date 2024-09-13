from .vector_store import get_vector_store_instance
from .custom_types import _EMBEDDING_TYPES, _VECTOR_DB
from typing import Optional, Any
from langchain_core.documents import Document
from .config import Config
from sqlalchemy import create_engine
import pandas as pd


def ingest_sql(
    file_path: str,
    description: str,
    embedding_model: _EMBEDDING_TYPES,
    index_name: str,
    dimension: Optional[int] = None,
    vector_db: _VECTOR_DB = "chromadb",
    **kwargs,
) -> Any:
    vector_store = get_vector_store_instance(
        embedding_model=embedding_model,
        index_name=index_name,
        dimension=dimension,
        vector_db=vector_db,
    )

    vector_store.add_documents(
        [Document(page_content=description, metadata={"search_type": "query"})]
    )

    engine = create_engine(f"{Config.POSTGRES_CONNECTION_STRING}/postgres")
    df = pd.read_excel(file_path, sheet_name=None)
    for sheet_name, df in df.items():
        table_name = sheet_name.replace(" ", "_").lower()
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
        print(f"Inserted data from sheet '{sheet_name}' into table '{table_name}'")

    return True
