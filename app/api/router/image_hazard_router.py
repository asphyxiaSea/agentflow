from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.api.models import FileItem
from app.core.errors import ExternalServiceError, InvalidRequestError
from app.workflows.image_hazard.graph import build_image_hazard_graph
from app.workflows.image_hazard.state import ImageHazardState

router = APIRouter(tags=["image hazard"])


@router.post("/image/hazard/detect")
async def detect_image_hazard(
    image_file: UploadFile = File(...),
    threshold: float = Form(0.25),
) -> dict[str, Any]:
    if not image_file.filename:
        raise InvalidRequestError(message="上传文件缺少文件名")
    content_type = image_file.content_type or ""
    if not content_type.startswith("image/"):
        raise InvalidRequestError(message="仅支持图片文件", detail={"content_type": content_type})

    try:
        content = await image_file.read()
        if not content:
            raise InvalidRequestError(message="图片内容为空")

        graph = build_image_hazard_graph()
        state: ImageHazardState = {
            "image_file_item": FileItem(
                filename=image_file.filename,
                content_type=content_type,
                data=content,
            ),
            "threshold": float(threshold),
        }
        result = await graph.ainvoke(state)
        return {
            "texts": result.get("texts", []),
            "text_groups": {
                "pest": result.get("pest_texts", []),
                "disease": result.get("disease_texts", []),
                "weed": result.get("weed_texts", []),
            },
            "sam_results": result.get("sam_results", []),
        }
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="虫病草检测失败", detail=str(exc)) from exc
    finally:
        await image_file.close()