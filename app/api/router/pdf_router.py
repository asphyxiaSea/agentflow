from __future__ import annotations

from typing import Any, List, Optional

from arq.jobs import Job, JobStatus
from fastapi import APIRouter, File, Form, Request, UploadFile

from app.api.schemas.pdf_schemas import (
    PdfParseResultResponse,
    SessionStatusResponse,
    SessionSubmitResponse,
)
from app.core.errors import AppError, ExternalServiceError, InvalidRequestError, SessionConflictError, SessionNotFoundError

router = APIRouter(tags=["files parse"])


@router.post("/files/parse", response_model=SessionSubmitResponse)
async def parse_pdf_to_structured(
    request: Request,
    session_id: str = Form(...),
    schema_model_json: str = Form(...),
    files: List[UploadFile] = File(...),
    system_prompt: Optional[str] = Form(None),
    pdf_process: Optional[str] = Form(None),
    text_process: Optional[str] = Form(None),
) -> SessionSubmitResponse:
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

        return SessionSubmitResponse(session_id=session_id, status=JobStatus.deferred.value)
    except InvalidRequestError:
        raise
    except AppError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="PDF 任务提交失败", detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()


@router.get("/files/parse/sessions/{session_id}", response_model=SessionStatusResponse)
async def parse_pdf_task_status(session_id: str, request: Request) -> SessionStatusResponse:
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


@router.get(
    "/files/parse/sessions/{session_id}/result",
    response_model=PdfParseResultResponse,
    response_model_exclude_none=True,
)
async def parse_pdf_task_result(session_id: str, request: Request) -> PdfParseResultResponse:
    redis = request.app.state.redis
    job = Job(session_id, redis)
    status = await job.status()

    if status == JobStatus.not_found:
        raise SessionNotFoundError(detail={"session_id": session_id})

    if status != JobStatus.complete:
        return PdfParseResultResponse(
            session_id=session_id, status=status.value, message="任务尚未完成"
        )

    info = await job.result_info()
    if info is None or not info.success:
        error = str(info.result) if info else "任务执行失败"
        return PdfParseResultResponse(session_id=session_id, status=status.value, error=error)

    result = info.result  # 就是 handler 返回的 {"results": ..., "extracted_texts": ...}
    return PdfParseResultResponse(
        session_id=session_id,
        status=status.value,
        results=result.get("results", []),
        extracted_texts=result.get("extracted_texts", []),
    )