import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# Text processing parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 150
MAX_TOKENS = 1500
TEMPERATURE = 0.7  # 0.1 -- 1.0

# ChromaDB Vector Database
COLLECTION_NAME = "rag_documents"
SIMILARITY_SEARCH_RESULTS = 3
DISTANCE_FUNCTION = "cosine"

# Web search
SERPER_API_URL = "https://google.serper.dev/search"
DEFAULT_SEARCH_RESULTS = 5

# Headers for web scraping
WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def validate_config():
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not found in environment variables")
        print("Please add OPENAI_API_KEY to your .env file")
        return False, None, None

    return True, OPENAI_API_KEY, SERPER_API_KEY
