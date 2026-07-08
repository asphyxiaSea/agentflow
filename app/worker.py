# app/worker.py
from __future__ import annotations

from arq.connections import RedisSettings

from app.application.pipelines.adaptive_rag_pipeline import (
    run_rag_chat_task,
    run_rag_chat_resume_task,
)

from app.application.pipelines.pdf_structured_pipeline import run_pdf_structured_task

class WorkerSettings:
    functions = [
        run_rag_chat_task,
        run_rag_chat_resume_task,
        run_pdf_structured_task,
    ]
    redis_settings = RedisSettings(host="localhost", port=6379)

    max_jobs = 4
    job_timeout = 300
    allow_abort_jobs = True
    # PDF 任务的业务结果直接存在 arq 里，需要比 RAG 任务留更长时间，
    # 具体多久取决于前端轮询 /result 的间隔和用户等待耐心，这里先给 30 分钟
    keep_result = 1800