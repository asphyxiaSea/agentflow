from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.workflows.adaptive_rag.nodes.common import current_question
from app.workflows.adaptive_rag.state import AdaptiveRagState


_ROUTER_PROMPT = (
    "你是工作流路由器，判断用户问题是否需要查询企业知识库。"
    "输出路由决策和原因。"
    "当前只有一个路由：agent_rag。"
)


class _RouteDecision(BaseModel):
    reason: str


async def route_decision_node(state: AdaptiveRagState) -> dict[str, Any]:
    decision = await get_chat_model().with_structured_output(_RouteDecision).ainvoke(
        [
            SystemMessage(content=_ROUTER_PROMPT),
            # 取最近八条信息给路由模型，让模型看到足够的上下文。
            *state["messages"][-8:],   
            HumanMessage(content=f"当前问题：{current_question(state)}"),
        ]
    )
    return {"route_reason": decision.reason.strip() or "无"}


def route_selector(state: AdaptiveRagState) -> str:
    return "agent_rag"