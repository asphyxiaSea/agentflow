from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import cast

from arq.connections import RedisSettings, create_pool
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router.pdf_structured_router import router as pdf_structured_router
from app.api.router.adaptive_rag_router import router as rag_router
from app.application.core.errors import AppError
from app.workflows.adaptive_rag.graph import create_adaptive_rag_graph

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


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
    app.state.redis = await create_pool(RedisSettings(host="localhost", port=6379))

    app.state.rag_graph, app.state.rag_saver  = await create_adaptive_rag_graph(REDIS_URL)
    


    try:
        yield
    finally:
        await app.state.redis.close()


def create_app() -> FastAPI:
    app = FastAPI(title="langchain app", lifespan=app_lifespan)
    app.add_exception_handler(AppError, app_error_handler)
    app.get("/health")(health)
    app.include_router(pdf_structured_router, prefix="/ai-workflow")
    app.include_router(rag_router, prefix="/ai-workflow")
    return app


app = create_app()