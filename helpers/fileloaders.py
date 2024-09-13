from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from typing import List


def load_csv(path: str) -> List[Document]:
    loader = CSVLoader(file_path=path)
    docs = loader.load()
    return docs


def load_excel(path: str) -> List[Document]:
    loader = UnstructuredExcelLoader(file_path=path, mode="elements")
    data = loader.load()
    print(data)
    docs = []
    for doc in data:
        metadata = {
            "source": f'Page: {doc.metadata["page_name"]} - {doc.metadata["source"]}'
        }
        rows = list(filter(None, doc.page_content.split("\n\n\n")))
        headers = rows.pop(0).split("\n")
        for row in rows:
            columns = row.split("\n")
            page_content = "\n".join(
                f"{header}: {column}" for header, column in zip(headers, columns)
            )
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs
