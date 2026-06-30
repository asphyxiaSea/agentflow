from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.errors import SessionNotFoundError
from app.core.settings import RAG_DEFAULT_KNOWLEDGE_DOMAIN
from app.workflows.adaptive_rag.graph import build_adaptive_rag_graph
from app.workflows.adaptive_rag.state import AdaptiveRagState, KbConfig

# ---------- payload schema ----------

class _UserMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1)


class RagChatPayload(BaseModel):
    session_id: str = Field(min_length=1)
    messages: list[_UserMessage] = Field(min_length=1)
    collection_name: str | None = None
    knowledge_domain: str = Field(default=RAG_DEFAULT_KNOWLEDGE_DOMAIN, min_length=1)
    book_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class ResumeRequest(BaseModel):
    decision: Literal["approve", "cancel"] = "approve"


class _ResumeTaskPayload(BaseModel):
    session_id: str = Field(min_length=1)
    decision: Literal["approve", "cancel"] = "approve"


# ---------- pipeline ----------


def _state_not_found(snapshot: Any) -> bool:
    return (
        not snapshot.values
        and not snapshot.next
        and snapshot.metadata is None
        and snapshot.created_at is None
    )


def _snapshot_to_public_state(session_id: str, snapshot: Any) -> dict[str, Any]:
    if _state_not_found(snapshot):
        raise SessionNotFoundError(detail={"session_id": session_id})

    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    interrupts = [interrupt.value for interrupt in snapshot.interrupts]
    next_nodes = list(snapshot.next)
    result = {
        "answer": values.get("answer", ""),
        "citations": values.get("citations", []),
    }

    return {
        "session_id": session_id,
        "next_nodes": next_nodes,
        "interrupts": interrupts,
        "result": result,
    }


def _snapshot_to_dispatcher_result(session_id: str, snapshot: Any) -> dict[str, Any]:
    public = _snapshot_to_public_state(session_id, snapshot)
    interrupts = public["interrupts"]
    if interrupts:
        first_interrupt = interrupts[0]
        interrupt_payload = first_interrupt if isinstance(first_interrupt, dict) else {"value": first_interrupt}
        return {
            "__task_status__": "INTERRUPTED",
            "interrupt_payload": interrupt_payload,
            "result": public["result"],
        }
    return public["result"]


def _build_thread_config(session_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": session_id.strip()}}


async def run_adaptive_rag_pipeline_start(payload: RagChatPayload) -> dict[str, Any]:
    """首次提交：跑到中断点或完成为止，不在这里等待人工输入。"""
    graph = build_adaptive_rag_graph()
    config = _build_thread_config(payload.session_id)

    messages: list[BaseMessage] = [
        HumanMessage(content=m.content.strip()) for m in payload.messages
    ]
    state: AdaptiveRagState = {
        "messages": messages,
        "kb_config": KbConfig(
            **{k: v for k, v in {
                "collection_name": payload.collection_name,
                "knowledge_domain": payload.knowledge_domain,
                "book_id": payload.book_id,
                "top_k": payload.top_k,
            }.items() if v}
        ),
    }

    await graph.ainvoke(state, config=config)
    return await get_adaptive_rag_session_state(payload.session_id)


async def run_adaptive_rag_pipeline_resume(session_id: str, decision: str) -> dict[str, Any]:
    """resume 续跑：用 Command(resume=...) 接着 checkpointer 里的状态继续执行。"""
    
    graph = build_adaptive_rag_graph()
    config = _build_thread_config(session_id)

    await graph.ainvoke(Command(resume=decision), config=config)
    return await get_adaptive_rag_session_state(session_id)


async def get_adaptive_rag_session_state(session_id: str) -> dict[str, Any]:
    graph = build_adaptive_rag_graph()
    snapshot = await graph.aget_state(_build_thread_config(session_id))
    return _snapshot_to_public_state(session_id.strip(), snapshot)


async def run_rag_chat_task(payload: dict[str, Any]) -> dict[str, Any]:
    rag_payload = RagChatPayload.model_validate(payload)

    graph = build_adaptive_rag_graph()
    config = _build_thread_config(rag_payload.session_id)
    messages: list[BaseMessage] = [
        HumanMessage(content=m.content.strip()) for m in rag_payload.messages
    ]
    state: AdaptiveRagState = {
        "messages": messages,
        "kb_config": KbConfig(
            **{k: v for k, v in {
                "collection_name": rag_payload.collection_name,
                "knowledge_domain": rag_payload.knowledge_domain,
                "book_id": rag_payload.book_id,
                "top_k": rag_payload.top_k,
            }.items() if v}
        ),
    }

    await graph.ainvoke(state, config=config)
    snapshot = await graph.aget_state(config)
    return _snapshot_to_dispatcher_result(rag_payload.session_id, snapshot)


async def run_rag_chat_resume_task(payload: dict[str, Any]) -> dict[str, Any]:
    resume_payload = _ResumeTaskPayload.model_validate(payload)

    graph = build_adaptive_rag_graph()
    config = _build_thread_config(resume_payload.session_id)
    await graph.ainvoke(Command(resume=resume_payload.decision), config=config)
    snapshot = await graph.aget_state(config)
    return _snapshot_to_dispatcher_result(resume_payload.session_id, snapshot)