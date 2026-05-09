from __future__ import annotations

from typing import Any, Literal, get_args

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.errors import InvalidRequestError
from app.core.model_factory import get_chat_model
from app.workflows.adaptive_rag.state import AdaptiveRagState, RagRoute


ADAPTIVE_RAG_ROUTER_PROMPT = (
    "你是工作流路由器。请在 direct_answer、fixed_rag、agent_rag、strict_insufficient 中选一个最合适的路由。"
)


class RouteDecision(BaseModel):
    route: RagRoute
    reason: str


_VALID_ROUTES: frozenset[str] = frozenset(get_args(RagRoute))


def current_question(state: AdaptiveRagState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = str(message.content).strip()
            if content:
                return content
    raise InvalidRequestError(message="messages 中缺少有效用户问题")


def conversation_preview(state: AdaptiveRagState, *, max_turns: int = 8) -> str:
    lines: list[str] = []
    for message in state["messages"][-max_turns:]:
        if isinstance(message, HumanMessage):
            lines.append(f"用户：{str(message.content).strip()}")
        elif isinstance(message, AIMessage):
            lines.append(f"助手：{str(message.content).strip()}")
        elif isinstance(message, SystemMessage):
            lines.append(f"系统：{str(message.content).strip()}")

    return "\n".join(line for line in lines if line.strip())


async def route_decision_node(state: AdaptiveRagState) -> dict[str, Any]:
    decision: RouteDecision = await get_chat_model().with_structured_output(RouteDecision).ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_ROUTER_PROMPT),
            HumanMessage(
                content=(
                    "请基于对话上下文判断路由策略。\n"
                    f"对话上下文：\n{conversation_preview(state)}\n\n"
                    f"当前问题：{current_question(state)}"
                )
            ),
        ]
    )

    return {
        "route": decision.route,
        "route_reason": decision.reason.strip() or "无",
    }


def route_selector(state: AdaptiveRagState) -> RagRoute:
    route = state.get("route")
    if route not in _VALID_ROUTES:
        raise InvalidRequestError(message="缺少有效路由结果", detail={"route": route})
    return route