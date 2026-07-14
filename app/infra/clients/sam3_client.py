from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.schema import FileItem
from app.core.errors import ExternalServiceError, InvalidRequestError


SAM3_SEMANTIC = "http://localhost:8002/sam3/image/segment/semantic/texts"
SAM3_INSTANCE = "http://localhost:8002/sam3/image/segment/instance/texts"


async def sam3_segment_semantic_texts(
    *,
    file_item: FileItem,
    config: dict[str, Any],
) -> dict[str, Any]:
    content_type = file_item.content_type or ""
    if not content_type.startswith("image/"):
        raise InvalidRequestError(message="不支持的图片类型", detail=content_type)

    if not file_item.data:
        raise InvalidRequestError(message="origin_file 内容为空")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                SAM3_SEMANTIC,
                files={
                    "image_file": (
                        file_item.filename,
                        file_item.data,
                        content_type,
                    )
                },
                data={"config": json.dumps(config, ensure_ascii=False)},
            )
            resp.raise_for_status()
            return resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise ExternalServiceError(message="Sam3 服务异常", detail=str(exc)) from exc


async def sam3_segment_instance_texts(
    *,
    file_item: FileItem,
    config: dict[str, Any],
) -> dict[str, Any]:
    content_type = file_item.content_type or ""
    if not content_type.startswith("image/"):
        raise InvalidRequestError(message="不支持的图片类型", detail=content_type)

    if not file_item.data:
        raise InvalidRequestError(message="origin_file 内容为空")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                SAM3_INSTANCE,
                files={
                    "image_file": (
                        file_item.filename,
                        file_item.data,
                        content_type,
                    )
                },
                data={"config": json.dumps(config, ensure_ascii=False)},
            )
            resp.raise_for_status()
            return resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise ExternalServiceError(message="Sam3 服务异常", detail=str(exc)) from exc


async def sam3_segment_geojson(*, file_item: FileItem, config: dict[str, Any]) -> dict[str, Any]:
    payload = await sam3_segment_semantic_texts(file_item=file_item, config=config)

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise InvalidRequestError(message="Sam3 返回结果为空", detail=payload)

    first_item = results[0] if isinstance(results[0], dict) else {}
    geojson = first_item.get("geojson")
    if not isinstance(geojson, dict):
        raise InvalidRequestError(message="Sam3 返回缺少 geojson", detail=payload)

    return geojson
