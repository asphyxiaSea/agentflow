from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class _ChatMessageIn(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1)


class RagChatRequest(BaseModel):
    """POST .../chat 的请求体（api 层契约，和 application 层的 RagChatPayload 各自独立定义）。"""

    messages: list[_ChatMessageIn] = Field(min_length=1)
    collection_name: str | None = None
    knowledge_domain: str | None = None
    book_id: str | None = None
    top_k: int | None = None


class RagResumeRequest(BaseModel):
    """POST .../resume 的请求体（api 层契约，对应 application 层的 ResumeTaskPayload）。"""

    decision: Literal["approve", "cancel"] = "approve"


class KbConfigRequest(BaseModel):
    """建会话时传入的检索范围配置。"""

    collection_name: str | None = None
    knowledge_domain: str | None = None
    book_id: str | None = None
    top_k: int | None = None


class SessionInitResponse(BaseModel):
    session_id: str
    kb_config: KbConfigRequest


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