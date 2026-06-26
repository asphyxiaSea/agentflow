from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.workflows.adaptive_rag.state import AdaptiveRagState


async def interrupt_node(state: AdaptiveRagState) -> dict[str, Any]:
    """在工具执行前暂停，将 tool_calls 暴露给调用方确认或修改。"""
    last_msg = state["messages"][-1]

    # 没有 tool_calls 就直接透传，不打断
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    # interrupt() 会暂停图执行，payload 就是调用方收到的内容
    human_decision = interrupt({
        "tool_calls": last_msg.tool_calls,
        "message": "LLM 即将调用以上工具，请确认（approve）或取消（cancel）。",
    })

    # 调用方 resume 时传回的值
    if human_decision == "cancel":
        # 取消：直接写入拒绝答案，后续 finalize 会读取
        return {
            "answer": "用户已取消工具调用，操作终止。",
        }

    # approve 或任意其他值 → 继续执行
    return {}