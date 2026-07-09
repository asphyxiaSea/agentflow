from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from arq.jobs import Job, JobStatus

from app.application.pipelines.adaptive_rag_pipeline import get_rag_session_state
from app.application.core.errors import SessionConflictError, SessionNotFoundError

router = APIRouter(tags=["rag"])


@router.post("/rag/chat/sessions/{session_id}/chat")
async def rag_chat(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    payload = await request.json()

    job = await redis.enqueue_job("run_rag_chat_task", payload, session_id, _job_id=session_id)
    if job is None:
        raise SessionConflictError(detail={"session_id": session_id, "status": "already running"})

    return {"session_id": session_id, "status": JobStatus.deferred.value}


@router.get("/rag/chat/sessions/{session_id}/status")
async def rag_chat_session_status(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    status = await job.status()

    if status == JobStatus.not_found:
        raise SessionNotFoundError(detail={"session_id": session_id})

    error: str | None = None
    if status == JobStatus.complete:
        info = await job.result_info()
        if info is not None and not info.success:
            error = str(info.result)

    return {"session_id": session_id, "status": status.value, "error": error}


@router.post("/rag/chat/sessions/{session_id}/cancel")
async def rag_chat_cancel(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    aborted = await job.abort()  # 排队中直接摘除；执行中会 cancel 掉 handler 协程（依赖 allow_abort_jobs=True）
    if not aborted:
        raise SessionConflictError(detail={"session_id": session_id, "status": "not cancellable"})
    return {"session_id": session_id, "status": "aborted"}


@router.get("/rag/chat/sessions/{session_id}/result")
async def rag_chat_session_result(session_id: str) -> dict[str, Any]:
    return await get_rag_session_state(session_id)  # 完全不变，还是查 checkpointer


@router.post("/rag/chat/sessions/{session_id}/resume")
async def rag_chat_resume(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    payload = await request.json()
    job = await redis.enqueue_job("run_rag_chat_resume_task", payload, session_id, _job_id=session_id)
    if job is None:
        raise SessionConflictError(detail={"session_id": session_id, "status": "still running or result not yet expired"})
    return {"session_id": session_id, "status": JobStatus.deferred.value}