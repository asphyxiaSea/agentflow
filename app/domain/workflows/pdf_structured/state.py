from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import NotRequired


class PdfStructuredState(TypedDict):
    pdf_path: str
    schema_model: dict[str, Any]
    system_prompt: NotRequired[str]
    pdf_process: NotRequired[dict[str, Any]]
    text_process: NotRequired[dict[str, Any]]
    paddle_pipeline: NotRequired[str]
    is_en_retry: NotRequired[bool]
    retry_with_rec_en: NotRequired[bool]
    extracted_text: NotRequired[str]
    structured_output: NotRequired[dict[str, Any]]
