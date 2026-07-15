from __future__ import annotations

from typing import Any
from app.domain.workflows.adaptive_rag.state import KbConfig
from langchain_core.messages import AIMessage, SystemMessage

from app.core.model_factory import get_chat_model
from app.domain.workflows.adaptive_rag.nodes.rag_agent.rag_tool_node import rag_tools
from app.domain.workflows.adaptive_rag.state import AdaptiveRagState


_AGENT_SYSTEM_PROMPT = """你是企业知识库问答助手，可使用工具：rewrite_query、retrieve_context。
当前知识领域：{domain_text}

决策规则：
1. 明显跨域 → 直接回复"依据不足（问题与当前知识领域不匹配）"，不调用工具。
2. 问题清晰 → 直接调用 retrieve_context。
3. 问题模糊/指代不清 → 先 rewrite_query，再 retrieve_context。
4. 首次结果为 NO_CONTEXT 或证据弱 → 最多再执行一次"改写+检索"。
5. 回答须基于检索证据，禁止编造；证据不足时回复"依据不足"。
6. 最终答案末尾给出引用编号，如 [1][2]。"""



async def llm_call_node(state: AdaptiveRagState) -> dict[str, Any]:
    domain_text = str(state.get("kb_config", KbConfig()).knowledge_domain).strip() or "未指定领域"
    prompt = _AGENT_SYSTEM_PROMPT.format(domain_text=domain_text)

    model = get_chat_model().bind_tools(rag_tools)
    messages = [SystemMessage(content=prompt)] + state["messages"]
    ai_msg: AIMessage = await model.ainvoke(messages)

    return {"messages": [ai_msg]}


async def finalize_node(state: AdaptiveRagState) -> dict[str, Any]:
    answer = ""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            answer = content.strip()
            break
        if isinstance(content, list):
            parts = [
                item["text"].strip()
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
            ]
            if parts:
                answer = "\n".join(parts)
                break

    return {
        "answer": answer or "没有检索到相关资料，当前无法基于知识库给出可靠答案。",
        # citations 已由 retrieve_context 工具通过 Command 写入 State，无需再处理
    }

