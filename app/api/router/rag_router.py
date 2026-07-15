from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from arq.jobs import Job, JobStatus

from app.api.schemas.rag_schemas import (
    RagAnswer,
    RagSessionResultResponse,
    SessionCancelResponse,
    SessionStatusResponse,
    SessionSubmitResponse,
)
from app.core.errors import SessionConflictError, SessionNotFoundError

router = APIRouter(tags=["rag"])


def _snapshot_not_found(snapshot: Any) -> bool:
    return (
        not snapshot.values
        and not snapshot.next
        and snapshot.metadata is None
        and snapshot.created_at is None
    )


async def _get_rag_session_state(request: Request, session_id: str) -> RagSessionResultResponse:
    graph = request.app.state.rag_graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": session_id.strip()}})

    if _snapshot_not_found(snapshot):
        raise SessionNotFoundError(detail={"session_id": session_id})

    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    interrupts = [i.value for i in snapshot.interrupts]

    return RagSessionResultResponse(
        session_id=session_id,
        next_nodes=list(snapshot.next),
        interrupts=interrupts,
        result=RagAnswer(
            answer=values.get("answer", ""),
            citations=values.get("citations", []),
        ),
    )


@router.post("/rag/chat/sessions/{session_id}/chat", response_model=SessionSubmitResponse)
async def rag_chat(session_id: str, request: Request) -> SessionSubmitResponse:
    redis = request.app.state.redis
    payload = await request.json()

    job = await redis.enqueue_job("run_rag_chat_task", payload, session_id, _job_id=session_id)
    if job is None:
        raise SessionConflictError(detail={"session_id": session_id, "status": "already running"})

    return SessionSubmitResponse(session_id=session_id, status=JobStatus.deferred.value)


@router.get("/rag/chat/sessions/{session_id}/status", response_model=SessionStatusResponse)
async def rag_chat_session_status(session_id: str, request: Request) -> SessionStatusResponse:
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

    return SessionStatusResponse(session_id=session_id, status=status.value, error=error)


@router.post("/rag/chat/sessions/{session_id}/cancel", response_model=SessionCancelResponse)
async def rag_chat_cancel(session_id: str, request: Request) -> SessionCancelResponse:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    aborted = await job.abort()  # 排队中直接摘除；执行中会 cancel 掉 handler 协程（依赖 allow_abort_jobs=True）
    if not aborted:
        raise SessionConflictError(detail={"session_id": session_id, "status": "not cancellable"})
    return SessionCancelResponse(session_id=session_id, status="aborted")


@router.get("/rag/chat/sessions/{session_id}/result", response_model=RagSessionResultResponse)
async def rag_chat_session_result(session_id: str, request: Request) -> RagSessionResultResponse:
    return await _get_rag_session_state(request, session_id)


@router.post("/rag/chat/sessions/{session_id}/resume", response_model=SessionSubmitResponse)
async def rag_chat_resume(session_id: str, request: Request) -> SessionSubmitResponse:
    redis = request.app.state.redis
    payload = await request.json()
    job = await redis.enqueue_job("run_rag_chat_resume_task", payload, session_id, _job_id=session_id)
    if job is None:
        raise SessionConflictError(
            detail={"session_id": session_id, "status": "still running or result not yet expired"}
        )
    return SessionSubmitResponse(session_id=session_id, status=JobStatus.deferred.value)