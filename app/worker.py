# app/worker.py
from __future__ import annotations

from arq.connections import RedisSettings

from app.application.pipelines.adaptive_rag_pipeline import (
    run_rag_chat_task,
    run_rag_chat_resume_task,
)


class WorkerSettings:
    """命令行独立进程入口：`arq app.worker.WorkerSettings`"""
    functions = [run_rag_chat_task, run_rag_chat_resume_task]
    redis_settings = RedisSettings(host="localhost", port=6379)

    max_jobs = 4
    job_timeout = 300
    allow_abort_jobs = True
    keep_result = 5