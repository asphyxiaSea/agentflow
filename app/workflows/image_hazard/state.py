from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import NotRequired

from app.api.models import FileItem


class ImageHazardState(TypedDict):
    image_file_item: FileItem
    threshold: NotRequired[float]
    pest_texts: NotRequired[list[str]]
    disease_texts: NotRequired[list[str]]
    weed_texts: NotRequired[list[str]]
    texts: NotRequired[list[str]]
    sam_results: NotRequired[list[dict[str, Any]]]
