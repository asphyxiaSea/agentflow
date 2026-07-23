from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.domain.workflows.adaptive_rag.state import AdaptiveRagState


_ROUTER_PROMPT = (
    "你是工作流路由器，判断用户问题是否需要查询企业知识库。\n"
    "当前知识领域：{domain_text}\n"
    "输出路由决策和原因。\n"
    "路由规则：\n"
    "- llm_call：问题与当前知识领域相关，需要检索后回答。\n"
    "- direct_answer：问题与当前知识领域无关（闲聊、跨域），直接回答无需检索。"
)

class _RouteDecision(BaseModel):
    route: Literal["llm_call", "direct_answer"]
    reason: str


async def route_decision_node(state: AdaptiveRagState) -> dict[str, Any]:
    domain_text = state.get("knowledge_domain", "").strip() or "未指定领域"
    print(f"路由决策节点：当前知识领域：{domain_text}")
    prompt = _ROUTER_PROMPT.format(domain_text=domain_text)

    decision = await get_chat_model().with_structured_output(_RouteDecision).ainvoke(
        [
            SystemMessage(content=prompt),
            *state["messages"][-8:],
        ]
    )
    return {
        "route": decision.route,
        "route_reason": decision.reason.strip() or "无",
    }

def route_selector(state: AdaptiveRagState) -> str:
    return state.get("route") or "llm_call"