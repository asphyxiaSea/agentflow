from __future__ import annotations

from typing import Any, List, Optional

from arq.jobs import Job, JobStatus
from fastapi import APIRouter, File, Form, Request, UploadFile

from app.core.errors import AppError, ExternalServiceError, InvalidRequestError, SessionConflictError, SessionNotFoundError

router = APIRouter(tags=["files parse"])


@router.post("/files/parse")
async def parse_pdf_to_structured(
    request: Request,
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

        redis = request.app.state.redis
        job = await redis.enqueue_job(
            "run_pdf_structured_task",
            {
                "schema_model_json": schema_model_json,
                "system_prompt": system_prompt or "",
                "pdf_process": pdf_process,
                "text_process": text_process,
                "files": file_payloads,
            },
            session_id,
            _job_id=session_id,
        )
        if job is None:
            raise SessionConflictError(detail={"session_id": session_id, "status": "already running"})

        return {"session_id": session_id, "status": "PENDING"}
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
async def parse_pdf_task_status(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    status = await job.status()

    if status == JobStatus.not_found:
        raise SessionNotFoundError(detail={"session_id": session_id})

    status_map = {
        JobStatus.deferred: "PENDING",
        JobStatus.queued: "PENDING",
        JobStatus.in_progress: "RUNNING",
    }
    mapped_status = status_map.get(status, "RUNNING")

    if status == JobStatus.complete:
        info = await job.result_info()
        mapped_status = "SUCCESS" if info and info.success else "FAILED"

    return {"session_id": session_id, "status": mapped_status}


@router.get("/files/parse/sessions/{session_id}/result")
async def parse_pdf_task_result(session_id: str, request: Request) -> dict[str, Any]:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    status = await job.status()

    if status == JobStatus.not_found:
        raise SessionNotFoundError(detail={"session_id": session_id})

    if status in (JobStatus.deferred, JobStatus.queued, JobStatus.in_progress):
        return {"session_id": session_id, "status": "PENDING" if status != JobStatus.in_progress else "RUNNING",
                "message": "任务尚未完成"}

    info = await job.result_info()
    if info is None or not info.success:
        error = str(info.result) if info else "任务执行失败"
        return {"session_id": session_id, "status": "FAILED", "error": error}

    result = info.result  # 就是 handler 返回的 {"results": ..., "extracted_texts": ...}
    return {
        "session_id": session_id,
        "status": "SUCCESS",
        "results": result.get("results", []),
        "extracted_texts": result.get("extracted_texts", []),
    }