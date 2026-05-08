from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.core.model_factory import get_chat_model
from app.core.settings import ADAPTIVE_RAG_DIRECT_PROMPT
from app.workflows.adaptive_rag.nodes.common import (
    build_trace,
    extract_route_reason,
    messages_for_direct_answer,
)
from app.workflows.adaptive_rag.state import AdaptiveRagState


async def direct_answer_node(state: AdaptiveRagState) -> dict[str, Any]:
    model = get_chat_model()
    result = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_DIRECT_PROMPT),
            *messages_for_direct_answer(state),
        ]
    )
    reason = extract_route_reason(state)
    answer = str(result.content)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "citations": [],
        "retrieval_count": 0,
        "trace": build_trace(route="direct_answer", reason=reason, retrieval_count=0),
    }
