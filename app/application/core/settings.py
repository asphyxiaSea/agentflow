import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:8001")
PADDLE_EXTRACT_PATH_URL = os.environ.get(
    "PADDLE_EXTRACT_PATH_URL",
    "http://localhost:8003/paddle/pp-structurev3/predict_path",
)

DEFAULT_MODEL_NAME = "llama3.1:8b"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000

RAG_CHROMA_PERSIST_DIR = os.environ.get("RAG_CHROMA_PERSIST_DIR", "./chroma_db")
RAG_CHROMA_COLLECTION = os.environ.get("RAG_CHROMA_COLLECTION", "rag_default")
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "nomic-embed-text")
RAG_EMBEDDING_BASE_URL = os.environ.get("RAG_EMBEDDING_BASE_URL", OLLAMA_BASE_URL)
RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "800"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))
RAG_RETRIEVAL_TOP_K = int(os.environ.get("RAG_RETRIEVAL_TOP_K", "4"))
RAG_DEFAULT_KNOWLEDGE_DOMAIN = os.environ.get("RAG_DEFAULT_KNOWLEDGE_DOMAIN", "agriculture")

TASK_QUEUE_MAXSIZE = int(
    os.environ.get("TASK_QUEUE_MAXSIZE", os.environ.get("RAG_TASK_QUEUE_MAXSIZE", "100"))
)
TASK_WORKER_COUNT = int(
    os.environ.get("TASK_WORKER_COUNT", os.environ.get("RAG_TASK_WORKER_COUNT", "5"))
)
TASK_TIMEOUT_SECONDS = float(
    os.environ.get("TASK_TIMEOUT_SECONDS", os.environ.get("RAG_TASK_TIMEOUT_SECONDS", "90"))
)
TASK_RESULT_TTL_SECONDS = int(
    os.environ.get(
        "TASK_RESULT_TTL_SECONDS",
        os.environ.get("RAG_TASK_RESULT_TTL_SECONDS", "600"),
    )
)
TASK_CLEANUP_INTERVAL_SECONDS = int(
    os.environ.get(
        "TASK_CLEANUP_INTERVAL_SECONDS",
        os.environ.get("RAG_TASK_CLEANUP_INTERVAL_SECONDS", "60"),
    )
)

# Backward-compatible aliases for existing imports.
RAG_TASK_QUEUE_MAXSIZE = TASK_QUEUE_MAXSIZE
RAG_TASK_WORKER_COUNT = TASK_WORKER_COUNT
RAG_TASK_TIMEOUT_SECONDS = TASK_TIMEOUT_SECONDS
RAG_TASK_RESULT_TTL_SECONDS = TASK_RESULT_TTL_SECONDS
RAG_TASK_CLEANUP_INTERVAL_SECONDS = TASK_CLEANUP_INTERVAL_SECONDS
