from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.domain.workflows.adaptive_rag.state import AdaptiveRagState, KbConfig


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

async def init_rag_session(graph: Any, session_id: str, kb_config: KbConfig) -> None:
    """建会话：把 kb_config 写入 checkpoint 一次，不经过任何节点执行。
    之后整个会话生命周期内，kb_config 不会再被覆盖（chat/resume 都不再携带这个字段）。
    """
    config = _build_thread_config(session_id)
    await graph.aupdate_state(config, {"kb_config": kb_config})


# ---------- task handlers ----------

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
    # 注意：这里不再携带 kb_config。kb_config 只在建会话接口里写入一次，
    # 此处传的 state 里没有这个 key，LangGraph 合并时不会碰它，
    # 之前已经写入 checkpoint 的 kb_config 会原样保留。
    state: AdaptiveRagState = {"messages": messages}

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