from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast

from arq.connections import RedisSettings, create_pool
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router.pdf_router import router as pdf_structured_router
from app.api.router.rag_router import router as rag_router
from app.core.errors import AppError
from app.core.settings import REDIS_URL
from app.core.graph_bootstrap import bootstrap_pdf_graph, bootstrap_rag_graph

load_dotenv()


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    app_exc = cast(AppError, exc)
    return JSONResponse(
        status_code=app_exc.status_code,
        content={
            "error": {
                "code": app_exc.code,
                "message": app_exc.message,
                "detail": app_exc.detail,
            }
        },
    )


async def health() -> dict[str, str]:
    return {"status": "ok"}


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    app.state.redis = await create_pool(RedisSettings.from_dsn(REDIS_URL))

    app.state.rag_graph, app.state.rag_saver = await bootstrap_rag_graph(REDIS_URL)
    app.state.pdf_graph, app.state.pdf_saver = await bootstrap_pdf_graph(REDIS_URL)

    try:
        yield
    finally:
        rag_saver = cast(Any, getattr(app.state, "rag_saver", None))
        if rag_saver is not None and hasattr(rag_saver, "aclose"):
            await rag_saver.aclose()

        pdf_saver = cast(Any, getattr(app.state, "pdf_saver", None))
        if pdf_saver is not None and hasattr(pdf_saver, "aclose"):
            await pdf_saver.aclose()
        await app.state.redis.close()


def create_app() -> FastAPI:
    app = FastAPI(title="agent app", lifespan=app_lifespan)
    app.add_exception_handler(AppError, app_error_handler)
    app.get("/health")(health)
    app.include_router(pdf_structured_router, prefix="/ai-workflow")
    app.include_router(rag_router, prefix="/ai-workflow")
    return app

app = create_app()