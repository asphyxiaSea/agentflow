from __future__ import annotations

import asyncio
from typing import Annotated, Any

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.infra.clients.chroma_client import search_chroma, build_citations
from app.domain.workflows.adaptive_rag.state import AdaptiveRagState, KbConfig


class _Rewrite(BaseModel):
    rewritten_query: str


async def _rewrite(question: str) -> str:
    result: _Rewrite = await get_chat_model().with_structured_output(_Rewrite).ainvoke(
        [SystemMessage(content="将用户问题改写为适合向量检索的单句查询，保留实体名与关键约束。"), HumanMessage(content=question)]
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

def _build_context_blocks(docs_with_scores: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{i}] source={doc.metadata.get('source', 'unknown')}, "
        f"chunk={doc.metadata.get('chunk_index', -1)}, score={score:.4f}\n{doc.page_content}"
        for i, (doc, score) in enumerate(docs_with_scores, 1)
    )

@tool
async def rewrite_query(query: str) -> str:
    """将用户问题改写为适合向量检索的单句查询。"""
    return await _rewrite(query)


@tool
async def retrieve_context(
    query: str,
    state: Annotated[AdaptiveRagState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],  # ToolNode 自动注入
) -> Command:
    """检索知识库中与查询相关的文档片段。"""
    docs = await _retrieve(query, state)
    citations = build_citations(docs) if docs else []
    context = _build_context_blocks(docs) if docs else "NO_CONTEXT"

    return Command(update={
        "citations": citations,           # 直接写回 State
        "messages": [ToolMessage(
            content=context,              # 这是 LLM 看到的内容
            tool_call_id=tool_call_id,
        )],
    })


rag_tools = [rewrite_query, retrieve_context]
rag_tool_node = ToolNode(rag_tools)