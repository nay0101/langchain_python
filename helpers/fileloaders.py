from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from typing import List
import nltk


def loadCSV(path: str) -> List[Document]:
    loader = CSVLoader(file_path=path)
    docs = loader.load()
    return docs


def loadExcel(path: str) -> List[Document]:
    nltk.download("punkt_tab")
    loader = UnstructuredExcelLoader(file_path=path)
    data = loader.load()
    docs = []
    for doc in data:
        metadata = doc.metadata
        rows = list(filter(None, doc.page_content.split("\n\n\n")))
        headers = rows.pop(0).split("\n")
        for row in rows:
            columns = row.split("\n")
            page_content = "\n".join(
                f"{header}: {column}" for header, column in zip(headers, columns)
            )
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs
