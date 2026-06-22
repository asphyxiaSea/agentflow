from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import InjectedState, ToolNode
from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.infra.clients.chroma_client import build_citations, search_chroma
from app.workflows.adaptive_rag.nodes.common import build_context_blocks
from app.workflows.adaptive_rag.state import AdaptiveRagState, KbConfig


_REWRITE_PROMPT = "将用户问题改写为适合向量检索的单句查询，保留实体名与关键约束。"


class _Rewrite(BaseModel):
    rewritten_query: str


async def _rewrite(question: str) -> str:
    result: _Rewrite = await get_chat_model().with_structured_output(_Rewrite).ainvoke(
        [SystemMessage(content=_REWRITE_PROMPT), HumanMessage(content=question)]
    )
    return result.rewritten_query.strip() or question


async def _retrieve(query: str, state: AdaptiveRagState) -> list[tuple[Document, float]]:
    cfg = state.get("kb_config") or KbConfig()
    filter_map = {
        k: v for k, v in {
            "domain": cfg.knowledge_domain,
            "book_id": cfg.book_id,
        }.items() if v is not None
    }
    return await asyncio.to_thread(
        search_chroma,
        query=query.strip(),
        top_k=cfg.top_k,
        collection_name=cfg.collection_name,
        metadata_filter=filter_map or None,
    )


@tool
async def rewrite_query(query: str) -> str:
    """将用户问题改写为适合向量检索的单句查询。"""
    return await _rewrite(query)


@tool
async def retrieve_context(
    query: str,
    state: Annotated[AdaptiveRagState, InjectedState],
) -> str:
    """检索知识库中与查询相关的文档片段。"""
    docs = await _retrieve(query, state)
    return build_context_blocks(docs) if docs else "NO_CONTEXT"


rag_tools = [rewrite_query, retrieve_context]
rag_tool_node = ToolNode(rag_tools)