from __future__ import annotations

import json
import os
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import InvalidRequestError
from app.core.schema import FileItem

from app.domain.workflows.pdf_structured.state import PdfStructuredState


# ---------- payload schema ----------

class PdfStructuredPayload(BaseModel):
    schema_model_json: str = Field(min_length=1)
    system_prompt: str = ""
    pdf_process: str | None = None
    text_process: str | None = None
    files: list[FileItem] = Field(min_length=1)

    def parsed_schema_model(self) -> dict[str, Any]:
        try:
            result = json.loads(self.schema_model_json)
        except Exception as exc:
            raise InvalidRequestError(message="Invalid schema_model JSON", detail=str(exc)) from exc
        if not isinstance(result, dict):
            raise InvalidRequestError(message="schema_model 必须是 JSON object")
        return result

    def parsed_pdf_process(self) -> dict[str, Any] | None:
        return self._parse_optional_json(self.pdf_process, "pdf_process")

    def parsed_text_process(self) -> dict[str, Any] | None:
        return self._parse_optional_json(self.text_process, "text_process")

    @staticmethod
    def _parse_optional_json(value: str | None, field: str) -> dict[str, Any] | None:
        if not value:
            return None
        try:
            result = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InvalidRequestError(message=f"Invalid {field} JSON", detail=str(exc)) from exc
        if not isinstance(result, dict):
            raise InvalidRequestError(message=f"{field} 必须是 JSON object")
        return result


# ---------- pipeline ----------

async def run_pdf_structured_pipeline(payload: PdfStructuredPayload, graph: Any) -> dict[str, Any]:
    schema_model = payload.parsed_schema_model()
    pdf_process = payload.parsed_pdf_process()
    text_process = payload.parsed_text_process()

    results: list[dict[str, Any]] = []
    extracted_texts: list[str] = []
    temp_paths: list[str] = []

    try:
        for file in payload.files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.data)
                temp_paths.append(tmp.name)

            state: PdfStructuredState = {
                "pdf_path": tmp.name,
                "schema_model": schema_model,
                "system_prompt": payload.system_prompt,
            }
            if pdf_process is not None:
                state["pdf_process"] = pdf_process
            if text_process is not None:
                state["text_process"] = text_process

            result = await graph.ainvoke(state)
            results.append(result.get("structured_output", {}))
            extracted_texts.append(result.get("extracted_text", ""))
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

    return {"results": results, "extracted_texts": extracted_texts}


# ---------- task handler ----------

async def run_pdf_structured_task(
    ctx: dict,
    payload: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """PDF 结构化抽取任务的 arq 入口。
    session_id 对这个任务本身没用（不像 RAG 任务要用它做 thread_id），
    只是 arq 调度层统一传参的一部分，这里直接忽略即可。
    """
    _ = session_id
    graph = ctx["pdf_graph"]
    return await run_pdf_structured_pipeline(PdfStructuredPayload.model_validate(payload), graph)