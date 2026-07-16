from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class KbConfigPayload(BaseModel):
    """建会话时传入的检索范围配置，会话生命周期内固定不变。"""

    collection_name: str | None = None
    knowledge_domain: str | None = None
    book_id: str | None = None
    top_k: int | None = None


class SessionInitResponse(BaseModel):
    """POST .../sessions/{session_id} 建会话的返回，把实际生效的 kb_config 回显给前端"""

    session_id: str
    kb_config: KbConfigPayload


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