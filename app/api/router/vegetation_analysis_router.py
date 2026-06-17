from __future__ import annotations

import json
import os
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.api.models import FileItem
from app.core.errors import ExternalServiceError, InvalidRequestError
from app.workflows.vegetation_analysis.graph import build_vegetation_analysis_graph
from app.workflows.vegetation_analysis.state import VegetationAnalysisState

router = APIRouter(tags=["vegetation analysis"])


def _parse_config(config_json: str) -> dict[str, Any]:
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(message="config_json 不是合法 JSON", detail=str(exc)) from exc

    if not isinstance(config, dict):
        raise InvalidRequestError(message="config_json 必须是 JSON object")
    if (texts := config.get("texts")) is not None and not isinstance(texts, list):
        raise InvalidRequestError(message="config_json.texts 必须是数组")
    if (threshold := config.get("threshold")) is not None and not isinstance(threshold, (int, float)):
        raise InvalidRequestError(message="config_json.threshold 必须是数字")

    return config


async def _read_image_file(file: UploadFile, field_name: str) -> FileItem:
    if not file.filename:
        raise InvalidRequestError(message="上传文件缺少文件名", detail={"field": field_name})
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise InvalidRequestError(
            message="仅支持图片文件",
            detail={"field": field_name, "content_type": content_type},
        )
    content = await file.read()
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    return FileItem(
        filename=file.filename,
        content_type=content_type,
        data=content,
        path=tmp_path,
    )


@router.post("/vegetation/analyze")
async def analyze_vegetation(
    config_json: str = Form(...),
    origin_file: UploadFile = File(...),
    ndvi_file: UploadFile = File(...),
    gndvi_file: UploadFile = File(...),
    lci_file: UploadFile = File(...),
) -> dict[str, Any]:
    upload_files = {
        "origin_file": origin_file,
        "ndvi_file": ndvi_file,
        "gndvi_file": gndvi_file,
        "lci_file": lci_file,
    }
    file_items: dict[str, FileItem] = {}

    try:
        config = _parse_config(config_json)

        for field_name, file in upload_files.items():
            file_items[field_name] = await _read_image_file(file, field_name)

        for field_name, item in file_items.items():
            if not item.path:
                raise InvalidRequestError(message="临时文件路径缺失", detail={"field": field_name})

        graph = build_vegetation_analysis_graph()
        state: VegetationAnalysisState = {
            "origin_file_item": file_items["origin_file"],
            "ndvi_file_item": file_items["ndvi_file"],
            "gndvi_file_item": file_items["gndvi_file"],
            "lci_file_item": file_items["lci_file"],
            "config": config,
        }
        result = await graph.ainvoke(state)
        return {
            "geojson": result.get("geojson", {}),
            "index_stats": {
                "NDVI": result.get("ndvi_stats", {}),
                "GNDVI": result.get("gndvi_stats", {}),
                "LCI": result.get("lci_stats", {}),
            },
        }
    except InvalidRequestError:
        raise
    except Exception as exc:
        raise ExternalServiceError(message="植被分析失败", detail=str(exc)) from exc
    finally:
        for file in upload_files.values():
            await file.close()
        for item in file_items.values():
            if item.path and os.path.exists(item.path):
                os.remove(item.path)