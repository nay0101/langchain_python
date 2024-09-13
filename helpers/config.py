import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
    HUGGINGFACEHUB_API_TOKEN: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ELASTIC_API_KEY: str | None = os.getenv("ELASTIC_API_KEY")
    ELASTIC_CLOUD_ID: str | None = os.getenv("ELASTIC_CLOUD_ID")
    COHERE_API_KEY: str | None = os.getenv("COHERE_API_KEY")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASEURL: str | None = os.getenv("LANGFUSE_BASEURL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    POSTGRES_CONNECTION_STRING: str | None = os.getenv("POSTGRES_CONNECTION_STRING")
