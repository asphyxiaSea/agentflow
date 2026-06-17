from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.core.settings import RAG_CHROMA_COLLECTION, RAG_RETRIEVAL_TOP_K
from app.infra.clients.chroma_client import build_citations, search_chroma
from app.workflows.adaptive_rag.nodes.common import (
    build_context_blocks,
    current_question,
)
from app.workflows.adaptive_rag.state import AdaptiveRagState, KbConfig


# ---------- prompts ----------

_AGENT_SYSTEM_PROMPT = """你是企业知识库问答助手，可使用工具：rewrite_query、retrieve_context。
当前知识领域：{domain_text}

决策规则：
1. 明显跨域 → 直接回复"依据不足（问题与当前知识领域不匹配）"，不调用工具。
2. 问题清晰 → 直接调用 retrieve_context。
3. 问题模糊/指代不清 → 先 rewrite_query，再 retrieve_context。
4. 首次结果为 NO_CONTEXT 或证据弱 → 最多再执行一次"改写+检索"。
5. 回答须基于检索证据，禁止编造；证据不足时回复"依据不足"。
6. 最终答案末尾给出引用编号，如 [1][2]。"""

_REWRITE_PROMPT = "将用户问题改写为适合向量检索的单句查询，保留实体名与关键约束。"
_GRADE_DOC_PROMPT = "判断以下文本是否包含回答该问题所需的信息，只回答 yes 或 no。"
_GRADE_ANSWER_PROMPT = (
    "判断该答案是否基于上下文信息回答了问题且未编造事实。"
    "满足则回答 pass，否则回答 retry。"
)

_MAX_TOOL_ROUNDS = 4


# ---------- schemas ----------

class _Rewrite(BaseModel):
    rewritten_query: str

# ---------- runtime ----------

@dataclass
class _AgentRuntime:
    rewritten_question: str
    answer: str = ""
    docs_with_scores: list[tuple[Document, float]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0


# ---------- helpers ----------

async def _rewrite(question: str) -> str:
    result: _Rewrite = await get_chat_model().with_structured_output(_Rewrite).ainvoke(
        [SystemMessage(content=_REWRITE_PROMPT), HumanMessage(content=question)]
    )
    return result.rewritten_query.strip() or question


async def _retrieve(query: str, state: AdaptiveRagState) -> list[tuple[Document, float]]:
    cfg = state.get("kb_config") or KbConfig()
    filter_map = {
        k: v for k, v in {
            "domain": cfg.knowledge_domain,
            "book_id": cfg.book_id,
        }.items() if v
    }
    return await asyncio.to_thread(
        search_chroma,
        query=query.strip(),
        top_k=cfg.top_k,
        collection_name=cfg.collection_name,
        metadata_filter=filter_map or None,
    )


# ---------- tools ----------

def _make_tools(state: AdaptiveRagState, rt: _AgentRuntime, question: str):
    @tool
    async def rewrite_query(query: str) -> str:
        """Rewrite user query into a concise retrieval-friendly query."""
        rt.rewritten_question = await _rewrite(query)
        return rt.rewritten_question

    @tool
    async def retrieve_context(query: str) -> str:
        """Retrieve and grade relevant knowledge-base chunks for a query."""
        docs = await _retrieve(query, state)
        rt.docs_with_scores = docs
        rt.citations = build_citations(docs) if docs else []
        rt.retrieval_count += 1
        return build_context_blocks(docs) if docs else "NO_CONTEXT"

    tools = [rewrite_query, retrieve_context]
    return tools, {t.name: t for t in tools}


# ---------- agent runner ----------

async def _run_agent(
    state: AdaptiveRagState,
    domain_text: str,
    extra_instruction: str = "",
) -> _AgentRuntime:
    question = current_question(state)
    rt = _AgentRuntime(rewritten_question=question)
    tools, tools_by_name = _make_tools(state, rt, question)

    model = get_chat_model().bind_tools(tools)

    prompt = _AGENT_SYSTEM_PROMPT.format(domain_text=domain_text)
    if extra_instruction:
        prompt += f"\n\n补充：{extra_instruction}"

    messages: list[BaseMessage] = [
        SystemMessage(content=prompt),
        *state["messages"],
    ]

    for _ in range(_MAX_TOOL_ROUNDS):
        ai_msg: AIMessage = await model.ainvoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            break

        tool_results: list[ToolMessage] = []
        for tc in ai_msg.tool_calls:
            t = tools_by_name[tc["name"]]
            result = await t.ainvoke(tc["args"])
            tool_results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        messages.extend(tool_results)

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                rt.answer = content.strip()
                break
            if isinstance(content, list):
                parts = [
                    item["text"].strip()
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
                ]
                if parts:
                    rt.answer = "\n".join(parts)
                    break

    return rt


# ---------- node ----------

async def agent_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    domain_text = str(state.get("knowledge_domain") or "").strip() or "未指定领域"
    rt = await _run_agent(state, domain_text)
    answer = rt.answer or "没有检索到相关资料，当前无法基于知识库给出可靠答案。"

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "citations": rt.citations,
    }