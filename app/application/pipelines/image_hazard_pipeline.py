from __future__ import annotations

from typing import Any

from app.api.models import FileItem
from app.core.errors import InvalidRequestError
from app.workflows.image_hazard.graph import build_image_hazard_graph
from app.workflows.image_hazard.state import ImageHazardState


async def run_image_hazard_pipeline(
    *,
    image_file_item: FileItem,
    threshold: float = 0.25,
) -> dict[str, Any]:
    if not image_file_item.data:
        raise InvalidRequestError(message="图片内容为空")

    graph = build_image_hazard_graph()
    state: ImageHazardState = {
        "image_file_item": image_file_item,
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
