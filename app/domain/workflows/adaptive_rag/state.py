from __future__ import annotations
import operator

from typing import Any, Annotated, TypedDict
from typing_extensions import NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class AdaptiveRagState(TypedDict):
    # ---- 对话内容 ----
    messages: Annotated[list[BaseMessage], add_messages]

    # ---- 检索配置：决定去哪个知识库、哪个领域/书籍范围里检索，检索几条 ----
    collection_name: NotRequired[str]  # 默认 RAG_CHROMA_COLLECTION
    knowledge_domain: NotRequired[str]  # 拼进 llm 提示词，默认空串表示不限定领域
    book_id: NotRequired[str]  # 限定书籍范围，默认空串表示不限定
    top_k: NotRequired[int]  # 默认 RAG_RETRIEVAL_TOP_K

    # ---- 本轮/累计的回答与引用 ----
    answer: NotRequired[str]
    citations: NotRequired[list[dict[str, Any]]]

    # ---- 路由决策 ----
    route: NotRequired[str]
    route_reason: NotRequired[str]