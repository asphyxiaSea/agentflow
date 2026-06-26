from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.workflows.adaptive_rag.state import AdaptiveRagState


_ROUTER_PROMPT = (
    "你是工作流路由器，判断用户问题是否需要查询企业知识库。\n"
    "输出路由决策和原因。\n"
    "路由规则：\n"
    "- llm_call：问题与知识库领域相关，需要检索后回答。\n"
    "- direct_answer：问题与知识库无关（闲聊、跨域），直接回答无需检索。"
)


class _RouteDecision(BaseModel):
    route: Literal["llm_call", "direct_answer"]
    reason: str


async def route_decision_node(state: AdaptiveRagState) -> dict[str, Any]:
    decision = await get_chat_model().with_structured_output(_RouteDecision).ainvoke(
        [
            SystemMessage(content=_ROUTER_PROMPT),
            *state["messages"][-8:],
        ]
    )
    return {
        "route": decision.route,
        "route_reason": decision.reason.strip() or "无",
    }


def route_selector(state: AdaptiveRagState) -> str:
    return state.get("route") or "llm_call"