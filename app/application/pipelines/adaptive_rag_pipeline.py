from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.settings import RAG_DEFAULT_KNOWLEDGE_DOMAIN
from app.domain.workflows.adaptive_rag.state import AdaptiveRagState, KbConfig


# ---------- payload schema ----------

class _UserMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1)


class RagChatPayload(BaseModel):
    messages: list[_UserMessage] = Field(min_length=1)
    collection_name: str | None = None
    knowledge_domain: str = Field(default=RAG_DEFAULT_KNOWLEDGE_DOMAIN, min_length=1)
    book_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class ResumeTaskPayload(BaseModel):
    decision: Literal["approve", "cancel"] = "approve"


# ---------- LangGraph session state ----------

def _build_thread_config(session_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": session_id.strip()}}


# ---------- task handlers ----------

# app/application/pipelines/adaptive_rag_pipeline.py

async def run_rag_chat_task(ctx: dict, payload: dict[str, Any], session_id: str) -> bool:
    """首次提交：跑到中断点或完成为止。
    中断状态/执行结果由 API 层统一查询 LangGraph checkpointer 获取。
    """
    rag_payload = RagChatPayload.model_validate(payload)
    graph = ctx["rag_graph"]
    config = _build_thread_config(session_id)

    messages: list[BaseMessage] = [
        HumanMessage(content=m.content.strip()) for m in rag_payload.messages
    ]
    state: AdaptiveRagState = {
        "messages": messages,
        "kb_config": KbConfig(**{k: v for k, v in {
            "collection_name": rag_payload.collection_name,
            "knowledge_domain": rag_payload.knowledge_domain,
            "book_id": rag_payload.book_id,
            "top_k": rag_payload.top_k,
        }.items() if v}),
    }

    await graph.ainvoke(state, config=config)
    snapshot = await graph.aget_state(config)
    return bool(snapshot.next)


async def run_rag_chat_resume_task(ctx: dict, payload: dict[str, Any], session_id: str) -> bool:
    """resume 续跑：用 Command(resume=...) 接着 checkpointer 里的状态继续执行。"""
    resume_payload = ResumeTaskPayload.model_validate(payload)
    graph = ctx["rag_graph"]
    config = _build_thread_config(session_id)

    await graph.ainvoke(Command(resume=resume_payload.decision), config=config)
    snapshot = await graph.aget_state(config)
    return bool(snapshot.next)