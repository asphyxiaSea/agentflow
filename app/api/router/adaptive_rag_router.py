from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.application.task_dispatcher import TaskType, get_task_dispatcher_service
from app.application.pipelines.adaptive_rag_pipeline import (
    RagChatPayload,
    ResumeRequest,
)
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["rag"])


@router.post("/rag/chat")
async def rag_chat(request: Request) -> dict[str, Any]:
    try:
        payload = RagChatPayload.model_validate(await request.json())
        dispatcher = get_task_dispatcher_service()
        session_id = await dispatcher.submit_task(
            task_type=TaskType.RAG_CHAT,
            session_id=payload.session_id,
            payload=payload.model_dump(),
        )
        return {"session_id": session_id, "status": "PENDING"}
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 会话启动失败", detail=str(exc)) from exc


@router.get("/rag/chat/sessions/{session_id}/state")
async def rag_chat_session_state(session_id: str) -> dict[str, Any]:
    try:
        dispatcher = get_task_dispatcher_service()
        return await dispatcher.get_task_snapshot(session_id)
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 会话状态查询失败", detail=str(exc)) from exc


@router.post("/rag/chat/sessions/{session_id}/resume")
async def rag_chat_resume(session_id: str, request: Request) -> dict[str, Any]:
    try:
        resume_payload = ResumeRequest.model_validate(await request.json())
        dispatcher = get_task_dispatcher_service()
        resumed_session_id = await dispatcher.resume_by_session(
            session_id=session_id,
            payload={
                "session_id": session_id.strip(),
                "decision": resume_payload.decision,
            },
        )
        return {"session_id": resumed_session_id, "status": "PENDING"}
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 会话恢复失败", detail=str(exc)) from exc