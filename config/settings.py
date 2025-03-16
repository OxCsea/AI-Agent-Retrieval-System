import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # ChromaDB配置
    COLLECTION_NAME = "ai_agents"
    PERSIST_DIR = "./chroma_db"
