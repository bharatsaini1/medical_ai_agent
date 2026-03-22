# utils/config.py
# ============================================================
#  Centralized configuration loader
#  Reads all values from .env and provides typed access
#  throughout the project — no os.getenv() scattered everywhere
# ============================================================

import os
from dotenv import load_dotenv

# Load .env file into process environment
load_dotenv()


class Config:
    """
    Single source of truth for all configuration values.
    Raises clear errors if required keys are missing.
    """

    # --- API Keys ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # --- Pinecone ---
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "medical-rag-index")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "384"))

    # --- LLM ---
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Llama 3.3 via Groq (latest, best quality)
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1                # Low temp = factual, deterministic

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast, lightweight, 384-dim

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.5"))
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))

    # --- Web Search ---
    WEB_SEARCH_RESULTS: int = int(os.getenv("WEB_SEARCH_RESULTS", "3"))

    # --- Data Paths ---
    DATA_PATH: str = os.path.join(os.path.dirname(__file__), "..", "data", "Diseases_Symptoms.csv")
    EMBEDDINGS_PATH: str = os.path.join(os.path.dirname(__file__), "..", "embeddings", "embeddings.npy")
    DOCUMENTS_PATH: str = os.path.join(os.path.dirname(__file__), "..", "embeddings", "documents.json")

    # --- Flask ---
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.path.join(os.path.dirname(__file__), "..", "logs", "app.log")

    @classmethod
    def validate(cls):
        """
        Call this at startup to catch missing API keys early
        instead of failing deep inside a request handler.
        """
        missing = []
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not cls.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")

        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please copy .env.example to .env and fill in your keys."
            )
        return True


# Singleton instance used across the project
config = Config()
