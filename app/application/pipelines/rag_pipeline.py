from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.domain.workflows.adaptive_rag.state import AdaptiveRagState


# ---------- payload schema ----------

class _UserMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1)


class RagChatPayload(BaseModel):
    """会话建立之后，每轮聊天只传消息本身。
    检索范围（collection_name/knowledge_domain/book_id/top_k）在建会话时一次性确定，
    此处不再接收，避免中途被静默覆盖。"""

    messages: list[_UserMessage] = Field(min_length=1)


class ResumeTaskPayload(BaseModel):
    decision: Literal["approve", "cancel"] = "approve"


# ---------- LangGraph session state ----------

def _build_thread_config(session_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": session_id.strip()}}


# ---------- session lifecycle ----------

async def init_rag_session(
    graph: Any, session_id: str, kb_config_fields: dict[str, Any]
) -> dict[str, Any]:
    """建会话：把检索配置平铺写入 checkpoint 一次，不经过任何节点执行。
    之后整个会话生命周期内，这些字段不会再被覆盖（chat/resume 都不再携带）。
    """
    session_kb_config = {
        "collection_name": str(kb_config_fields["collection_name"]).strip(),
        "top_k": int(kb_config_fields["top_k"]),
    }

    knowledge_domain_raw = kb_config_fields.get("knowledge_domain")
    knowledge_domain = (
        str(knowledge_domain_raw).strip() if isinstance(knowledge_domain_raw, str) else ""
    )
    if knowledge_domain:
        session_kb_config["knowledge_domain"] = knowledge_domain

    book_id_raw = kb_config_fields.get("book_id")
    book_id = str(book_id_raw).strip() if isinstance(book_id_raw, str) else ""
    if book_id:
        session_kb_config["book_id"] = book_id

    config = _build_thread_config(session_id)
    await graph.aupdate_state(config, session_kb_config)
    return session_kb_config


# ---------- task handlers ----------

async def run_rag_chat_task(ctx: dict, payload: dict[str, Any], session_id: str) -> str:
    """首次提交：跑到中断点或完成为止。
    中断状态/执行结果由 API 层统一查询 LangGraph checkpointer 获取。
    """
    rag_payload = RagChatPayload.model_validate(payload)
    graph = ctx["rag_graph"]
    config = _build_thread_config(session_id)

    messages: list[BaseMessage] = [
        HumanMessage(content=m.content.strip()) for m in rag_payload.messages
    ]
    # 检索配置字段只在建会话接口里写入一次，后续 chat 不再携带，
    # LangGraph 会保留 checkpoint 里已存在的 collection_name/knowledge_domain/book_id/top_k。
    state: AdaptiveRagState = {"messages": messages}

    await graph.ainvoke(state, config=config)
    snapshot = await graph.aget_state(config)
    return "interrupted" if snapshot.next else "completed"

async def run_rag_chat_resume_task(ctx: dict, payload: dict[str, Any], session_id: str) -> bool:
    """resume 续跑：用 Command(resume=...) 接着 checkpointer 里的状态继续执行。"""
    resume_payload = ResumeTaskPayload.model_validate(payload)
    graph = ctx["rag_graph"]
    config = _build_thread_config(session_id)

    await graph.ainvoke(Command(resume=resume_payload.decision), config=config)
    snapshot = await graph.aget_state(config)
    return bool(snapshot.next)