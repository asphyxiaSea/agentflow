from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from app.core.model_factory import get_chat_model
from app.workflows.adaptive_rag.state import AdaptiveRagState


_DIRECT_ANSWER_PROMPT = (
    "你是企业知识库问答助手。"
    "当前问题与知识库领域无关，直接根据自身知识回答即可，无需检索。"
)


async def direct_answer_node(state: AdaptiveRagState) -> dict[str, Any]:
    model = get_chat_model()
    messages = [SystemMessage(content=_DIRECT_ANSWER_PROMPT)] + state["messages"]
    ai_msg: AIMessage = await model.ainvoke(messages)
    answer = str(ai_msg.content).strip()

    return {
        "answer": answer,
        "messages": [ai_msg],
        "citations": [],
    }