import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
    ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_BASEURL = os.getenv("LANGFUSE_BASEURL")
