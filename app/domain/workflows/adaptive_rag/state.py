from __future__ import annotations
import operator

from typing import Any, Annotated, TypedDict
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.core.settings import RAG_CHROMA_COLLECTION, RAG_RETRIEVAL_TOP_K


class KbConfig(BaseModel):
    """会话级检索配置。仅在会话创建时写入一次，会话生命周期内不再变更。"""

    collection_name: str = RAG_CHROMA_COLLECTION
    knowledge_domain: str = ""
    book_id: str = ""
    top_k: int = RAG_RETRIEVAL_TOP_K



class AdaptiveRagState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    kb_config: NotRequired[KbConfig]
    answer: NotRequired[str]
    citations: NotRequired[Annotated[list[dict[str, Any]], operator.add]]  # 改成 reducer
    route: NotRequired[str]
    route_reason: NotRequired[str]