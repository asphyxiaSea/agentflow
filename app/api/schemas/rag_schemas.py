from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SessionSubmitResponse(BaseModel):
    """POST .../chat 和 POST .../resume 的返回"""

    session_id: str
    status: str


class SessionCancelResponse(BaseModel):
    session_id: str
    status: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    error: str | None = None


class RagAnswer(BaseModel):
    answer: str
    citations: list[dict[str, Any]]


class RagSessionResultResponse(BaseModel):
    session_id: str
    next_nodes: list[str]
    interrupts: list[Any]
    result: RagAnswer