from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from typing import List, Optional

from app.application.task_dispatcher import TaskType, get_task_dispatcher_service
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError

router = APIRouter(tags=["files parse"])


@router.post("/files/parse")
async def parse_pdf_to_structured(
    session_id: str = Form(...),
    schema_model_json: str = Form(...),
    files: List[UploadFile] = File(...),
    system_prompt: Optional[str] = Form(None),
    pdf_process: Optional[str] = Form(None),
    text_process: Optional[str] = Form(None),
) -> dict[str, Any]:
    try:
        file_payloads: list[dict[str, Any]] = []
        for file in files:
            content = await file.read()
            file_payloads.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "data": content,
            })

        dispatcher = get_task_dispatcher_service()
        submitted_session_id = await dispatcher.submit_task(
            task_type=TaskType.PDF_STRUCTURED,
            session_id=session_id,
            payload={
                "schema_model_json": schema_model_json,
                "system_prompt": system_prompt or "",
                "pdf_process": pdf_process,
                "text_process": text_process,
                "files": file_payloads,
            },
        )
        return {"session_id": submitted_session_id, "status": "PENDING"}
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="PDF 任务提交失败", detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()


@router.get("/files/parse/sessions/{session_id}")
async def parse_pdf_task_status(session_id: str) -> dict[str, Any]:
    dispatcher = get_task_dispatcher_service()
    task = await dispatcher.get_task_snapshot(session_id)
    if task.get("task_type") != TaskType.PDF_STRUCTURED:
        raise InvalidRequestError(message="任务类型不匹配")
    return task


@router.get("/files/parse/sessions/{session_id}/result")
async def parse_pdf_task_result(session_id: str) -> dict[str, Any]:
    dispatcher = get_task_dispatcher_service()
    task = await dispatcher.get_task_snapshot(session_id)
    if task.get("task_type") != TaskType.PDF_STRUCTURED:
        raise InvalidRequestError(message="任务类型不匹配")

    status = task["status"]
    if status in ("PENDING", "RUNNING"):
        return {"session_id": session_id, "status": status, "message": "任务尚未完成"}
    if status == "FAILED":
        return {"session_id": session_id, "status": status, "error": task.get("error", "任务执行失败")}

    result = task.get("result") or {}
    return {
        "session_id": session_id,
        "status": status,
        "results": result.get("results", []),
        "extracted_texts": result.get("extracted_texts", []),
    }