import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class configs:
    # API Keys
    GROQ_API_KEY_1 = st.secrets["groq_api_key_1"]
    GROQ_API_KEY = st.secrets["groq_api_key"]
    QDRANT_API_KEY = st.secrets["qdrant_api_key"]
    QDRANT_URL = st.secrets["qdrant_url"]
    LLAMA_CLOUD_API_KEY = st.secrets["llama_cloud_api_key"]
    OPENAI_API_KEY = st.secrets.get("openai_api_key", "")

    # Collection names
    ARABIC_COLLECTION = "arabic_docs_llamaindex"
    ENGLISH_COLLECTION = "english_docs_llamaindex"

    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    GROQ_MODEL = "llama3-70b-8192"

    # Processing parameters
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    DEFAULT_TOP_K = 5
    SIMILARITY_CUTOFF = 0.3
    PAGES_TO_CHECK = 3
