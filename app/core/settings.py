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
ADAPTIVE_RAG_ROUTER_PROMPT = os.environ.get(
    "ADAPTIVE_RAG_ROUTER_PROMPT",
    "你是工作流路由器。请在 direct_answer、fixed_rag、agent_rag、strict_insufficient 中选一个最合适的路由。",
)
ADAPTIVE_RAG_DIRECT_PROMPT = os.environ.get(
    "ADAPTIVE_RAG_DIRECT_PROMPT",
    "你是企业知识助手。仅在不依赖外部事实或用户请求的是通用解释时直接作答，避免编造具体事实。",
)
ADAPTIVE_RAG_REWRITE_PROMPT = os.environ.get(
    "ADAPTIVE_RAG_REWRITE_PROMPT",
    "你是检索查询改写助手。请将用户问题改写为更适合向量检索的单句查询，保留实体名与关键约束。",
)

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
