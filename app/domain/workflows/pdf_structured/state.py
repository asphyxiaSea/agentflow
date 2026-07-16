from __future__ import annotations

from typing import Any, TypedDict
from typing_extensions import NotRequired


class PdfStructuredState(TypedDict):
    pdf_path: str
    schema_model: dict[str, Any]
    system_prompt: NotRequired[str]
    pdf_process: NotRequired[dict[str, Any]]
    text_process: NotRequired[dict[str, Any]]
    extracted_text: NotRequired[str]
    structured_output: NotRequired[dict[str, Any]]
