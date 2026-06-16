from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.model_factory import get_chat_model
from app.infra.clients.sam3_client import sam3_segment_instance_texts
from app.workflows.image_hazard.state import ImageHazardState


class HazardDetectResult(BaseModel):
    pest: list[str] = Field(default_factory=list)
    disease: list[str] = Field(default_factory=list)
    weed: list[str] = Field(default_factory=list)


def _normalize_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


async def llm_hazard_detect_node(state: ImageHazardState) -> dict[str, Any]:
    image_file_item = state["image_file_item"]
    image_base64 = base64.b64encode(image_file_item.data).decode("utf-8")
    mime = image_file_item.content_type or "image/jpeg"

    model = get_chat_model(model_name="gemma3:12b").with_structured_output(HazardDetectResult)
    result: HazardDetectResult = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are an agricultural vision assistant. "
                    "Detect potential pest, disease, and weed categories from the image. "
                    "Return only concise lowercase English terms."
                )
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Analyze this field image and return possible categories in three lists: "
                            "pest, disease, weed."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_base64}"},
                    },
                ]
            ),
        ]
    )

    pest_texts = _normalize_texts(result.pest)
    disease_texts = _normalize_texts(result.disease)
    weed_texts = _normalize_texts(result.weed)
    all_texts = _normalize_texts([*pest_texts, *disease_texts, *weed_texts])
    print(f"LLM Hazard Detection Result: pest={pest_texts}, disease={disease_texts}, weed={weed_texts}")

    # Ensure SAM receives at least one class prompt.
    if not all_texts:
        all_texts = ["pest", "disease", "weed"]

    return {
        "pest_texts": pest_texts,
        "disease_texts": disease_texts,
        "weed_texts": weed_texts,
        "texts": all_texts,
    }


async def sam_detect_node(state: ImageHazardState) -> dict[str, Any]:
    texts = state.get("texts") or []
    threshold = float(state.get("threshold", 0.25))

    sam_response = await sam3_segment_instance_texts(
        file_item=state["image_file_item"],
        config={
            "texts": texts,
            "threshold": threshold,
            "return_geojson": False,
        },
    )

    results = sam_response.get("results")
    return {
        "sam_results": results if isinstance(results, list) else [],
    }
