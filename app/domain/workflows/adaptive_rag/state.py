from __future__ import annotations
import operator

from dataclasses import dataclass
from typing import Any, Annotated, TypedDict
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.core.settings import RAG_CHROMA_COLLECTION, RAG_RETRIEVAL_TOP_K


@dataclass
class KbConfig:
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