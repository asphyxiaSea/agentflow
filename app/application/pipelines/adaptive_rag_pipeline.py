from __future__ import annotations

from typing import Any, Literal

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
    session_id: str = Field(min_length=1)
    decision: Literal["approve", "cancel"] = "approve"


# ---------- pipeline ----------

def _to_pipeline_output(result: dict[str, Any]) -> dict[str, Any]:
    if "__interrupt__" in result:
        return {"__interrupt__": result["__interrupt__"][0].value}
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }


async def run_adaptive_rag_pipeline_start(payload: RagChatPayload) -> dict[str, Any]:
    """首次提交：跑到中断点或完成为止，不在这里等待人工输入。"""
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
    return _to_pipeline_output(result)


async def run_adaptive_rag_pipeline_resume(session_id: str, decision: str) -> dict[str, Any]:
    """resume 续跑：用 Command(resume=...) 接着 checkpointer 里的状态继续执行。"""
    from langgraph.types import Command

    graph = build_adaptive_rag_graph()
    config: RunnableConfig = {"configurable": {"thread_id": session_id.strip()}}

    result = await graph.ainvoke(Command(resume=decision), config=config)
    return _to_pipeline_output(result)


# ---------- task handlers ----------

async def run_rag_chat_task(
    payload: dict[str, Any],
    task_record: TaskRecord,
    dispatcher: TaskDispatcherService,
) -> dict[str, Any]:
    rag_payload = RagChatPayload.model_validate(payload)

    result = await run_adaptive_rag_pipeline_start(rag_payload)

    if "__interrupt__" in result:
        await dispatcher._mark_interrupted(task_record.task_id, result["__interrupt__"])
        return {"__pending_interrupt__": True}

    return result


async def run_rag_chat_resume_task(
    payload: dict[str, Any],
    task_record: TaskRecord,
    dispatcher: TaskDispatcherService,
) -> dict[str, Any]:
    # 校验放在最前面：session_id 和 decision 都由前端直接提供，
    # 不再依赖查旧 TaskRecord，dispatcher 退化为纯任务执行器
    resume_payload = ResumeRequest.model_validate(payload)

    result = await run_adaptive_rag_pipeline_resume(
        resume_payload.session_id.strip(),
        resume_payload.decision,
    )

    if "__interrupt__" in result:
        await dispatcher._mark_interrupted(task_record.task_id, result["__interrupt__"])
        return {"__pending_interrupt__": True}

    return result