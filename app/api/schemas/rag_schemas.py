from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KbConfigPayload(BaseModel):
    """建会话时传入的检索范围配置，会话生命周期内固定不变。全部字段必填，
    避免请求体为空/不完整时被静默地用默认值建会话。"""

    collection_name: str = Field(min_length=1)
    knowledge_domain: str = Field(min_length=1)
    book_id: str = Field(min_length=1)
    top_k: int = Field(gt=0)


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