from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from arq.jobs import Job, JobStatus

from app.application.pipelines.adaptive_rag_pipeline import get_rag_session_state
from app.core.errors import SessionConflictError, SessionNotFoundError

router = APIRouter(tags=["rag"])

_STATUS_MAP = {
    JobStatus.deferred: "PENDING",
    JobStatus.queued: "PENDING",
    JobStatus.in_progress: "RUNNING",
    JobStatus.not_found: None,
}


@router.post("/rag/chat/sessions/{session_id}/chat")
async def rag_chat(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    payload = await request.json()

    # _job_id=session_id：arq 内部用这个 key 做去重，
    # 如果同一个 job_id 还在排队/执行中，enqueue_job 直接返回 None，天然替代了原来的 SessionConflictError 判断
    job = await redis.enqueue_job("run_rag_chat_task", payload, session_id, _job_id=session_id)
    if job is None:
        raise SessionConflictError(detail={"session_id": session_id, "status": "already running"})

    return {"session_id": session_id, "status": "PENDING"}


@router.get("/rag/chat/sessions/{session_id}/status")
async def rag_chat_session_status(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    status = await job.status()

    if status == JobStatus.not_found:
        raise SessionNotFoundError(detail={"session_id": session_id})

    mapped_status = _STATUS_MAP.get(status, "RUNNING")
    error: str | None = None

    if status == JobStatus.complete:
        info = await job.result_info()
        if info is None:
            mapped_status = "SUCCESS"
        elif not info.success:
            mapped_status = "FAILED"
            error = str(info.result)
        else:
            mapped_status = "INTERRUPTED" if info.result else "SUCCESS"

    return {"session_id": session_id, "status": mapped_status, "error": error}


@router.post("/rag/chat/sessions/{session_id}/cancel")
async def rag_chat_cancel(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    aborted = await job.abort()  # 排队中直接摘除；执行中会 cancel 掉 handler 协程（依赖 allow_abort_jobs=True）
    if not aborted:
        raise SessionConflictError(detail={"session_id": session_id, "status": "not cancellable"})
    return {"session_id": session_id, "status": "CANCELED"}


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
    return {"session_id": session_id, "status": "PENDING"}