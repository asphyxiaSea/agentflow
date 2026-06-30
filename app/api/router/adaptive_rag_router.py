from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.application.task_dispatcher import TaskType, get_task_dispatcher_service
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError, TaskNotFoundError


router = APIRouter(tags=["rag"])


@router.post("/rag/chat")
async def rag_chat(request: Request) -> dict[str, Any]:
    try:
        dispatcher = get_task_dispatcher_service()
        task_id = await dispatcher.submit_task(
            task_type=TaskType.RAG_CHAT,
            payload=await request.json(),
        )
        return {"task_id": task_id, "status": "PENDING"}
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 任务提交失败", detail=str(exc)) from exc


@router.get("/rag/chat/tasks/{task_id}")
async def rag_chat_task_status(task_id: str) -> dict[str, Any]:
    dispatcher = get_task_dispatcher_service()
    return await dispatcher.get_task_snapshot(task_id)


@router.get("/rag/chat/tasks/{task_id}/result")
async def rag_chat_task_result(task_id: str) -> dict[str, Any]:
    dispatcher = get_task_dispatcher_service()
    task = await dispatcher.get_task_snapshot(task_id)
    status = task["status"]

    if status in ("PENDING", "RUNNING"):
        return {"task_id": task_id, "status": status, "message": "任务尚未完成"}

    if status == "INTERRUPTED":
        return {
            "task_id": task_id,
            "status": status,
            "interrupt_payload": task.get("interrupt_payload"),
        }

    if status == "FAILED":
        return {"task_id": task_id, "status": status, "error": task.get("error", "任务执行失败")}

    result = task.get("result") or {}
    return {
        "task_id": task_id,
        "status": status,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }


@router.post("/rag/chat/{task_id}/resume")
async def rag_chat_resume(task_id: str, request: Request) -> dict[str, Any]:
    try:
        dispatcher = get_task_dispatcher_service()
        await dispatcher.resume_task(task_id, payload=await request.json())
        return {"task_id": task_id, "status": "PENDING"}
    except InvalidRequestError:
        raise
    except TaskNotFoundError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="RAG 任务恢复失败", detail=str(exc)) from exc