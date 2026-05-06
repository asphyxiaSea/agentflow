from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from app.core.errors import InvalidRequestError, PermissionDeniedError
from app.workflows.adaptive_rag.graph import build_adaptive_rag_graph
from app.workflows.adaptive_rag.state import AdaptiveRagState


async def _authorize_rag_thread_owner(*, graph: Any, thread_id: str, user_id: str) -> None:
    snapshot = await graph.aget_state(
        {
            "configurable": {
                "thread_id": thread_id,
            }
        }
    )
    snapshot_config = snapshot.config if isinstance(snapshot.config, dict) else {}
    raw_configurable = snapshot_config.get("configurable")
    snapshot_configurable: dict[str, Any] = (
        raw_configurable if isinstance(raw_configurable, dict) else {}
    )
    snapshot_user_id = snapshot_configurable.get("user_id")

    # Allow first-time thread usage; enforce identity once owner is recorded.
    if snapshot_user_id is None:
        return
    if not isinstance(snapshot_user_id, str) or not snapshot_user_id.strip():
        raise InvalidRequestError(message="thread_id 快照中的 user_id 非法")
    if snapshot_user_id.strip() != user_id:
        raise PermissionDeniedError(message="thread_id 不属于当前 user_id")


async def run_adaptive_rag_pipeline(
    *,
    messages: list[dict[str, Any]],
    thread_id: str,
    user_id: str,
    collection_name: str | None = None,
    knowledge_domain: str | None = None,
    book_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    normalized_thread_id = thread_id.strip()
    normalized_user_id = user_id.strip()

    if not messages:
        raise InvalidRequestError(message="messages 不能为空")

    normalized_messages: list[BaseMessage] = [
        HumanMessage(content=str(message["content"]).strip())
        for message in messages
        if str(message["content"]).strip()
    ]
    if not normalized_messages:
        raise InvalidRequestError(message="messages 不能为空")

    graph = build_adaptive_rag_graph()
    await _authorize_rag_thread_owner(
        graph=graph,
        thread_id=normalized_thread_id,
        user_id=normalized_user_id,
    )

    state: AdaptiveRagState = {
        "messages": normalized_messages,
    }
    if collection_name is not None:
        state["collection_name"] = collection_name
    if knowledge_domain is not None:
        state["knowledge_domain"] = knowledge_domain
    if book_id is not None:
        state["book_id"] = book_id
    if top_k is not None:
        state["top_k"] = top_k

    result = await graph.ainvoke(
        state,
        config={
            "configurable": {
                "thread_id": normalized_thread_id,
                "user_id": normalized_user_id,
            }
        },
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "trace": result.get("trace", {}),
    }
