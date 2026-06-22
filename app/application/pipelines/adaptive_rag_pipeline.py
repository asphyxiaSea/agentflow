from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.core.settings import RAG_DEFAULT_KNOWLEDGE_DOMAIN
from app.workflows.adaptive_rag.graph import build_adaptive_rag_graph
from app.workflows.adaptive_rag.state import AdaptiveRagState, KbConfig


# ---------- payload schema ----------

class _UserMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1)


class RagChatPayload(BaseModel):
    session_id: str = Field(min_length=1)
    messages: list[_UserMessage] = Field(min_length=1)
    collection_name: str | None = None
    knowledge_domain: str = Field(default=RAG_DEFAULT_KNOWLEDGE_DOMAIN, min_length=1)
    book_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


# ---------- pipeline ----------

async def run_adaptive_rag_pipeline(payload: RagChatPayload) -> dict[str, Any]:
    graph = build_adaptive_rag_graph()

    messages: list[BaseMessage] = [
        HumanMessage(content=m.content.strip()) for m in payload.messages
    ]

    state: AdaptiveRagState = {
        "messages": messages,
        "kb_config": KbConfig(
            **{k: v for k, v in {
                "collection_name": payload.collection_name,
                "knowledge_domain": payload.knowledge_domain,
                "book_id": payload.book_id,
                "top_k": payload.top_k,
            }.items() if v}
        ),
    }

    result = await graph.ainvoke(
        state,
        config={
            "configurable": {
                "thread_id": payload.session_id.strip(),
            }
        },
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }


# ---------- task handler ----------

async def run_rag_chat_task(payload: dict[str, Any]) -> dict[str, Any]:
    return await run_adaptive_rag_pipeline(RagChatPayload.model_validate(payload))