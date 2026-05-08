from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain.tools import tool

from app.core.model_factory import get_chat_model
from app.workflows.adaptive_rag.nodes.common import (
    answer_with_context,
    build_context_blocks,
    build_trace,
    conversation_preview,
    current_question,
    extract_final_text,
    extract_route_reason,
    messages_for_agent,
    retrieve,
    rewrite_query,
)
from app.workflows.adaptive_rag.state import AdaptiveRagState


async def agent_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    reason = extract_route_reason(state)
    question = current_question(state)
    dialogue_context = conversation_preview(state)
    knowledge_domain = str(state.get("knowledge_domain") or "").strip()
    domain_text = knowledge_domain or "未指定领域"

    runtime: dict[str, Any] = {
        "rewritten_question": question,
        "docs_with_scores": [],
        "citations": [],
        "retrieval_count": 0,
    }

    @tool
    async def rewrite_query_tool(query: str) -> str:
        """Rewrite user query into a concise retrieval-friendly query."""
        rewritten = await rewrite_query(query)
        runtime["rewritten_question"] = rewritten
        return rewritten

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant knowledge-base chunks for a query and return formatted context."""
        docs_with_scores, citations = retrieve(
            query=query,
            collection_name=state.get("collection_name"),
            knowledge_domain=state.get("knowledge_domain"),
            book_id=state.get("book_id"),
            top_k=state.get("top_k"),
        )
        runtime["docs_with_scores"] = docs_with_scores
        runtime["citations"] = citations
        runtime["retrieval_count"] = int(runtime["retrieval_count"]) + 1
        if not docs_with_scores:
            return "NO_CONTEXT"
        return build_context_blocks(docs_with_scores)

    model = get_chat_model()
    agent = create_agent(model=model, tools=[rewrite_query_tool, retrieve_context])
    agent_messages = messages_for_agent(state)
    if not agent_messages:
        agent_messages = [{"role": "user", "content": question}]

    agent_result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"你是企业知识库问答助手。你可以使用两个工具：rewrite_query 与 retrieve_context。"
                        f"当前知识领域约束为：{domain_text}。"
                        "你的目标是在保证事实可靠的前提下，尽量高效地回答。"
                        "决策规则如下："
                        "1) 先判断用户问题是否与当前知识领域一致。"
                        "2) 若与当前领域高度相关："
                        "2.1 问题表述清晰、实体明确、检索词充分，可直接调用 retrieve_context。"
                        "2.2 问题口语化、指代不清、多轮省略、关键词缺失，先调用 rewrite_query，再用改写结果调用 retrieve_context。"
                        "3) 若与当前领域差异过大或明显跨域：不要调用任何检索工具，直接回复“依据不足（问题与当前知识领域不匹配）”。"
                        "4) 若首次检索结果为 NO_CONTEXT 或证据弱，可执行一次“改写 + 再检索”。"
                        "5) 最多进行 2 次检索，避免无效循环。"
                        "6) 回答必须基于检索证据，禁止编造；证据不足时明确回复“依据不足”。"
                        "7) 最终答案精炼，并在末尾给出引用编号，如 [1][2]。"
                    ),
                },
                *agent_messages,
            ],
        },
        config={"recursion_limit": 8},
    )

    final_messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    answer = extract_final_text(final_messages)
    docs_with_scores = runtime["docs_with_scores"]
    citations = runtime["citations"]
    retrieval_count = int(runtime["retrieval_count"])
    rewritten_question = str(runtime["rewritten_question"])

    if not answer:
        if not docs_with_scores:
            answer = "没有检索到相关资料，当前无法基于知识库给出可靠答案。"
        else:
            contexts = build_context_blocks(docs_with_scores)
            answer = await answer_with_context(
                question=question,
                rewritten_question=rewritten_question,
                contexts=contexts,
                dialogue_context=dialogue_context,
            )

    return {
        "rewritten_question": rewritten_question,
        "docs_with_scores": docs_with_scores,
        "citations": citations,
        "retrieval_count": retrieval_count,
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "trace": build_trace(
            route="agent_rag",
            reason=reason,
            retrieval_count=retrieval_count,
        ),
    }
