from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel

from app.core.errors import InvalidRequestError
from app.core.model_factory import get_chat_model
from app.core.settings import RAG_CHROMA_COLLECTION, RAG_RETRIEVAL_TOP_K
from app.infra.clients.chroma_client import build_citations, search_chroma
from app.workflows.adaptive_rag.state import AdaptiveRagState


ADAPTIVE_RAG_REWRITE_PROMPT = (
    "你是检索查询改写助手。请将用户问题改写为更适合向量检索的单句查询，保留实体名与关键约束。"
)

_AGENT_SYSTEM_PROMPT = (
    "你是企业知识库问答助手。你可以使用两个工具：rewrite_query 与 retrieve_context。"
    "当前知识领域约束为：{domain_text}。"
    "你的目标是在保证事实可靠的前提下，尽量高效地回答。"
    "决策规则如下："
    "1) 先判断用户问题是否与当前知识领域一致。"
    "2) 若与当前领域高度相关："
    "2.1 问题表述清晰、实体明确、检索词充分，可直接调用 retrieve_context。"
    "2.2 问题口语化、指代不清、多轮省略、关键词缺失，先调用 rewrite_query，再用改写结果调用 retrieve_context。"
    "3) 若与当前领域差异过大或明显跨域：不要调用任何检索工具，直接回复\"依据不足（问题与当前知识领域不匹配）\"。"
    "4) 若首次检索结果为 NO_CONTEXT 或证据弱，可执行一次\"改写 + 再检索\"。"
    "5) 最多进行 2 次检索，避免无效循环。"
    "6) 回答必须基于检索证据，禁止编造；证据不足时明确回复\"依据不足\"。"
    "7) 最终答案精炼，并在末尾给出引用编号，如 [1][2]。"
)


class QueryRewrite(BaseModel):
    rewritten_query: str


def build_trace(*, route: str, reason: str, retrieval_count: int) -> dict[str, Any]:
    return {
        "route": route,
        "reason": reason,
        "retrieval_count": retrieval_count,
    }


def extract_route_reason(state: AdaptiveRagState) -> str:
    return str(state.get("route_reason") or "").strip() or "路由模型未提供原因"


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


def extract_final_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text
            elif isinstance(content, list):
                parts = [
                    item["text"].strip()
                    for item in content
                    if isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                    and item["text"].strip()
                ]
                if parts:
                    return "\n".join(parts)
    return ""


def build_context_blocks(docs_with_scores: list[tuple[Any, float]]) -> str:
    return "\n\n".join(
        f"[{idx}] source={doc.metadata.get('source', 'unknown')}, "
        f"chunk={doc.metadata.get('chunk_index', -1)}, score={score:.4f}\n{doc.page_content}"
        for idx, (doc, score) in enumerate(docs_with_scores, start=1)
    )


async def answer_with_context(
    *,
    question: str,
    rewritten_question: str,
    contexts: str,
    dialogue_context: str,
) -> str:
    result = await get_chat_model().ainvoke(
        [
            SystemMessage(content=(
                "你是企业知识库问答助手。"
                "只能根据给定上下文回答，禁止编造。"
                "若证据不足，请明确说明\"依据不足\"。"
                "答案请精炼，并在末尾给出引用编号，如 [1][2]。"
            )),
            HumanMessage(content=(
                f"多轮对话：\n{dialogue_context}\n\n"
                f"原始问题：{question}\n"
                f"改写检索问题：{rewritten_question}\n\n"
                f"可用上下文如下：\n{contexts}"
            )),
        ]
    )
    return str(result.content)


def retrieve(
    *,
    query: str,
    collection_name: str | None,
    knowledge_domain: str | None,
    book_id: str | None,
    top_k: int | None,
) -> tuple[list[tuple[Any, float]], list[dict[str, Any]]]:
    filter_map: dict[str, Any] = {}
    if domain := (knowledge_domain or "").strip():
        filter_map["domain"] = domain
    if book := (book_id or "").strip():
        filter_map["book_id"] = book

    docs_with_scores = search_chroma(
        query=query.strip(),
        top_k=top_k or RAG_RETRIEVAL_TOP_K,
        collection_name=collection_name or RAG_CHROMA_COLLECTION,
        metadata_filter=filter_map or None,
    )
    return docs_with_scores, build_citations(docs_with_scores)


async def rewrite_query(question: str) -> str:
    result: QueryRewrite = await get_chat_model().with_structured_output(QueryRewrite).ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
    )
    return result.rewritten_query.strip() or question.strip()


async def agent_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    reason = extract_route_reason(state)
    question = current_question(state)
    dialogue_context = conversation_preview(state)
    domain_text = str(state.get("knowledge_domain") or "").strip() or "未指定领域"

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
        return build_context_blocks(docs_with_scores) if docs_with_scores else "NO_CONTEXT"

    agent = create_agent(
        model=get_chat_model(),
        tools=[rewrite_query_tool, retrieve_context],
    )

    agent_result = await agent.ainvoke(
        {# type: ignore[arg-type]
            "messages": [
                SystemMessage(content=_AGENT_SYSTEM_PROMPT.format(domain_text=domain_text)),
                *state["messages"],
            ],
        },
        config={"recursion_limit": 8},
    )
    
    final_messages: list[BaseMessage] = agent_result.get("messages", [])
    answer = extract_final_text(final_messages)
    docs_with_scores = runtime["docs_with_scores"]
    citations = runtime["citations"]
    retrieval_count = int(runtime["retrieval_count"])
    rewritten_question = str(runtime["rewritten_question"])

    if not answer:
        answer = (
            "没有检索到相关资料，当前无法基于知识库给出可靠答案。"
            if not docs_with_scores
            else await answer_with_context(
                question=question,
                rewritten_question=rewritten_question,
                contexts=build_context_blocks(docs_with_scores),
                dialogue_context=dialogue_context,
            )
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