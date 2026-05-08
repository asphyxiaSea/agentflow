from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.workflows.adaptive_rag.nodes.common import build_trace, extract_route_reason
from app.workflows.adaptive_rag.state import AdaptiveRagState


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
