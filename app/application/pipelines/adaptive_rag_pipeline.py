from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.application.task_dispatcher import TaskDispatcherService, TaskRecord
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


# ---------- pipeline ----------

async def run_adaptive_rag_pipeline(
    payload: RagChatPayload,
    *,
    on_interrupt: Callable[[dict[str, Any]], Awaitable[str]],
) -> dict[str, Any]:
    from langgraph.types import Command

    graph = build_adaptive_rag_graph()
    config: RunnableConfig = {"configurable": {"thread_id": payload.session_id.strip()}}

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

    result = await graph.ainvoke(state, config=config)

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        decision = await on_interrupt(interrupt_payload)
        result = await graph.ainvoke(Command(resume=decision), config=config)

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }


# ---------- task handler ----------

async def run_rag_chat_task(
    payload: dict[str, Any],
    task_record: TaskRecord,
    dispatcher: TaskDispatcherService,
) -> dict[str, Any]:
    rag_payload = RagChatPayload.model_validate(payload)

    async def on_interrupt(interrupt_payload: dict[str, Any]) -> str:
        # 1. 标记 INTERRUPTED，存人工决策的 payload，供调用方 POST /resume 时使用
        await dispatcher._mark_interrupted(task_record.task_id, interrupt_payload)

        # 2. 挂 Event，等调用方 POST /resume
        event = asyncio.Event()
        async with dispatcher._task_lock:
            task_record.resume_event = event

        # 3. 阻塞等待唤醒
        await asyncio.wait_for(event.wait(), timeout=dispatcher._task_timeout_seconds)

        # 4. 校验在这里做，resume_value 是调用方透传的原始 dict
        body = ResumeRequest.model_validate(task_record.resume_value)
        return body.decision

    return await run_adaptive_rag_pipeline(rag_payload, on_interrupt=on_interrupt)