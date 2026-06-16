from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.api.models import FileItem
from app.application.pipelines.image_hazard_pipeline import run_image_hazard_pipeline
from app.core.errors import ExternalServiceError, InvalidRequestError


router = APIRouter(tags=["image hazard"])


def _validate_image_upload(file: UploadFile) -> None:
    if not file.filename:
        raise InvalidRequestError(message="上传文件缺少文件名")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise InvalidRequestError(message="仅支持图片文件", detail={"content_type": content_type})


@router.post("/image/hazard/detect")
async def detect_image_hazard(
    image_file: UploadFile = File(...),
    threshold: float = Form(0.25),
) -> dict[str, Any]:
    _validate_image_upload(image_file)

    try:
        content = await image_file.read()
        file_item = FileItem(
            filename=image_file.filename or "image.jpg",
            content_type=image_file.content_type or "application/octet-stream",
            data=content,
        )

        return await run_image_hazard_pipeline(
            image_file_item=file_item,
            threshold=threshold,
        )
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="虫病草检测失败", detail=str(exc)) from exc
    finally:
        await image_file.close()
