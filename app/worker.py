from __future__ import annotations

from arq.connections import RedisSettings

from app.application.pipelines.rag_pipeline import (
    run_rag_chat_task,
    run_rag_chat_resume_task,
)
from app.application.pipelines.pdf_pipeline import run_pdf_structured_task
from app.core.graph_bootstrap import bootstrap_pdf_graph, bootstrap_rag_graph
from app.core.settings import REDIS_URL


async def startup(ctx: dict) -> None:
    ctx["rag_graph"], ctx["rag_saver"] = await bootstrap_rag_graph(REDIS_URL)
    ctx["pdf_graph"], ctx["pdf_saver"] = await bootstrap_pdf_graph(REDIS_URL)

async def shutdown(ctx: dict) -> None:
    rag_saver = ctx.get("rag_saver")
    if rag_saver is not None and hasattr(rag_saver, "aclose"):
        await rag_saver.aclose()

    pdf_saver = ctx.get("pdf_saver")
    if pdf_saver is not None and hasattr(pdf_saver, "aclose"):
        await pdf_saver.aclose()


class WorkerSettings:
    functions = [
        run_rag_chat_task,
        run_rag_chat_resume_task,
        run_pdf_structured_task,
    ]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    on_startup = startup
    on_shutdown = shutdown

    max_jobs = 4
    job_timeout = 300
    allow_abort_jobs = True
    keep_result = 5