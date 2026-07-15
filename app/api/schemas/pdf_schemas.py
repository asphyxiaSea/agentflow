from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SessionSubmitResponse(BaseModel):
    """POST /files/parse 提交任务的返回"""

    session_id: str
    status: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    error: str | None = None


class PdfParseResultResponse(BaseModel):
    """GET /files/parse/sessions/{id}/result 的返回。

    同一个接口在不同状态下返回不同字段：
    - 未完成时：message 有值
    - 失败时：error 有值
    - 成功时：results / extracted_texts 有值
    配合 response_model_exclude_none=True，没用到的字段不会出现在响应体里。
    """

    session_id: str
    status: str
    message: str | None = None
    error: str | None = None
    results: list[dict[str, Any]] | None = None
    extracted_texts: list[str] | None = None