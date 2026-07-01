from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request


from app.application.task_dispatcher import TaskType, get_task_dispatcher_service
from app.application.pipelines.adaptive_rag_pipeline import (
    ResumeTaskPayload,
    get_rag_session_state,
)
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["rag"])


@router.post("/rag/chat/sessions/{session_id}/chat")
async def rag_chat(session_id: str, request: Request) -> dict[str, Any]:
    try:
        dispatcher = get_task_dispatcher_service()
        await dispatcher.submit_task(
            task_type=TaskType.RAG_CHAT,
            session_id=session_id,
            payload=await request.json(),
        )
        return {"session_id": session_id, "status": "PENDING"}
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 会话启动失败", detail=str(exc)) from exc


@router.get("/rag/chat/sessions/{session_id}/status")
async def rag_chat_session_status(session_id: str) -> dict[str, Any]:
    """查询 dispatcher 层的任务执行状态（PENDING/RUNNING/SUCCESS/FAILED）。
    前端用来判断任务是否跑完，跑完后再调用 /result 拿业务结果。
    """
    try:
        dispatcher = get_task_dispatcher_service()
        return await dispatcher.get_task_snapshot(session_id)
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="任务状态查询失败", detail=str(exc)) from exc


@router.get("/rag/chat/sessions/{session_id}/result")
async def rag_chat_session_result(session_id: str) -> dict[str, Any]:
    """查询 LangGraph checkpointer 里的业务结果，包含：
    - result: answer + citations（图执行完成时有值）
    - interrupts: 中断点信息（图被 interrupt() 暂停时有值）
    - next_nodes: 下一个待执行节点（非空说明图还没跑完）
    这是中断状态和最终答案的唯一权威来源，不经过 dispatcher。
    """
    try:
        return await get_rag_session_state(session_id)
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="会话结果查询失败", detail=str(exc)) from exc


@router.post("/rag/chat/sessions/{session_id}/resume")
async def rag_chat_resume(session_id: str, request: Request) -> dict[str, Any]:
    try:
        dispatcher = get_task_dispatcher_service()
        await dispatcher.resume_by_session(
            session_id=session_id,
            payload=await request.json(),
        )
        return {"session_id": session_id, "status": "PENDING"}
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 会话恢复失败", detail=str(exc)) from exc