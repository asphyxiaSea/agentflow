from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.errors import InvalidRequestError
from app.core.model_factory import get_chat_model
from app.core.settings import ADAPTIVE_RAG_ROUTER_PROMPT
from app.workflows.adaptive_rag.nodes.common import conversation_preview, current_question
from app.workflows.adaptive_rag.state import AdaptiveRagState, RagRoute


class RouteDecision(BaseModel):
    route: Literal[
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    ]
    reason: str


async def route_decision_node(state: AdaptiveRagState) -> dict[str, Any]:
    question = current_question(state)
    dialogue_context = conversation_preview(state)
    model = get_chat_model().with_structured_output(RouteDecision)
    decision_raw = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_ROUTER_PROMPT),
            HumanMessage(
                content=(
                    "请基于对话上下文判断路由策略。\n"
                    f"对话上下文：\n{dialogue_context}\n\n"
                    f"当前问题：{question}"
                )
            ),
        ]
    )

    if isinstance(decision_raw, RouteDecision):
        decision = decision_raw
    elif isinstance(decision_raw, BaseModel):
        decision = RouteDecision.model_validate(decision_raw.model_dump())
    elif isinstance(decision_raw, dict):
        decision = RouteDecision.model_validate(decision_raw)
    else:
        raise InvalidRequestError(message="LLM 路由结果格式非法", detail={"type": str(type(decision_raw))})

    route = str(decision.route)
    if route not in {
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    }:
        raise InvalidRequestError(message="LLM 路由结果非法", detail={"route": route})

    return {
        "route": route,
        "route_reason": decision.reason.strip() or "无",
    }


def route_selector(state: AdaptiveRagState) -> RagRoute:
    route = state.get("route")
    if route in {
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    }:
        return route
    raise InvalidRequestError(message="缺少有效路由结果", detail={"route": route})
