# app/main.py
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import cast

from arq.connections import RedisSettings, create_pool
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router.image_hazard_router import router as image_hazard_router
from app.api.router.vegetation_analysis_router import router as vegetation_analysis_router
from app.api.router.pdf_structured_router import router as pdf_structured_router
from app.api.router.adaptive_rag_router import router as rag_router
from app.core.errors import AppError

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
    app.state.redis = await create_pool(RedisSettings(host="localhost", port=6379))
    try:
        yield
    finally:
        await app.state.redis.close()


def create_app() -> FastAPI:
    app = FastAPI(title="langchain app", lifespan=app_lifespan)
    app.add_exception_handler(AppError, app_error_handler)
    app.get("/health")(health)
    app.include_router(pdf_structured_router, prefix="/ai-workflow")
    app.include_router(vegetation_analysis_router, prefix="/ai-workflow")
    app.include_router(image_hazard_router, prefix="/ai-workflow")
    app.include_router(rag_router, prefix="/ai-workflow")
    return app


app = create_app()