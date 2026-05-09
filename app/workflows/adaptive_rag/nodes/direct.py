from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.errors import InvalidRequestError
from app.core.model_factory import get_chat_model
from app.workflows.adaptive_rag.state import AdaptiveRagState


ADAPTIVE_RAG_DIRECT_PROMPT = (
    "你是企业知识助手。仅在不依赖外部事实或用户请求的是通用解释时直接作答，避免编造具体事实。"
)


def build_trace(*, route: str, reason: str, retrieval_count: int) -> dict[str, Any]:
    return {
        "route": route,
        "reason": reason,
        "retrieval_count": retrieval_count,
    }


def extract_route_reason(state: AdaptiveRagState) -> str:
    reason = str(state.get("route_reason") or "")
    return reason.strip() or "路由模型未提供原因"


def current_question(state: AdaptiveRagState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = str(message.content).strip()
            if content:
                return content
    raise InvalidRequestError(message="messages 中缺少有效用户问题")


def messages_for_direct_answer(state: AdaptiveRagState) -> list[BaseMessage]:
    turns: list[BaseMessage] = [
        message for message in state["messages"] if not isinstance(message, SystemMessage)
    ]
    if not turns:
        turns = [HumanMessage(content=current_question(state))]
    return turns


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
