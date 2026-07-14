from __future__ import annotations

import os

from arq.connections import RedisSettings

from app.application.pipelines.adaptive_rag_pipeline import (
    run_rag_chat_task,
    run_rag_chat_resume_task,
)
from app.application.pipelines.pdf_structured_pipeline import run_pdf_structured_task
from app.application.workflows.adaptive_rag.graph import create_adaptive_rag_graph

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


async def startup(ctx: dict) -> None:
    ctx["rag_graph"], ctx["rag_saver"] = await create_adaptive_rag_graph(REDIS_URL)

async def shutdown(ctx: dict) -> None:
    saver = ctx.get("rag_saver")
    if saver is not None and hasattr(saver, "aclose"):
        await saver.aclose()


class WorkerSettings:
    functions = [
        run_rag_chat_task,
        run_rag_chat_resume_task,
        run_pdf_structured_task,
    ]
    redis_settings = RedisSettings(host="localhost", port=6379)
    on_startup = startup
    on_shutdown = shutdown

    max_jobs = 4
    job_timeout = 300
    allow_abort_jobs = True
    keep_result = 5