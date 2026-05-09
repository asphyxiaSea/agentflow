from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.workflows.adaptive_rag.state import AdaptiveRagState


def build_trace(*, route: str, reason: str, retrieval_count: int) -> dict[str, Any]:
    return {
        "route": route,
        "reason": reason,
        "retrieval_count": retrieval_count,
    }


def extract_route_reason(state: AdaptiveRagState) -> str:
    reason = str(state.get("route_reason") or "")
    return reason.strip() or "路由模型未提供原因"


async def strict_insufficient_node(state: AdaptiveRagState) -> dict[str, Any]:
    reason = extract_route_reason(state)
    answer = "依据不足，当前无法给出可靠答案。"
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "citations": [],
        "retrieval_count": 0,
        "trace": build_trace(route="strict_insufficient", reason=reason, retrieval_count=0),
    }
