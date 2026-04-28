from __future__ import annotations

from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel

from app.core.model_factory import get_chat_model
from app.core.errors import InvalidRequestError
from app.core.settings import (
    ADAPTIVE_RAG_DIRECT_PROMPT,
    ADAPTIVE_RAG_REWRITE_PROMPT,
    ADAPTIVE_RAG_ROUTER_PROMPT,
    RAG_CHROMA_COLLECTION,
    RAG_RETRIEVAL_TOP_K,
)
from app.infra.clients.chroma_client import build_citations, search_chroma
from app.workflows.adaptive_rag.state import AdaptiveRagState, RagRoute


class RouteDecision(BaseModel):
    route: Literal[
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    ]
    reason: str


class QueryRewrite(BaseModel):
    rewritten_query: str


def _build_trace(*, route: str, reason: str, retrieval_count: int) -> dict[str, Any]:
    return {
        "route": route,
        "reason": reason,
        "retrieval_count": retrieval_count,
    }


def _extract_route_reason(state: AdaptiveRagState) -> str:
    reason = str(state.get("route_reason") or "")
    return reason.strip() or "路由模型未提供原因"


def _current_question(state: AdaptiveRagState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = str(message.content).strip()
            if content:
                return content
    raise InvalidRequestError(message="messages 中缺少有效用户问题")


def _conversation_preview(state: AdaptiveRagState, *, max_turns: int = 8) -> str:
    lines: list[str] = []
    for message in state["messages"][-max_turns:]:
        if isinstance(message, HumanMessage):
            lines.append(f"用户：{str(message.content).strip()}")
        elif isinstance(message, AIMessage):
            lines.append(f"助手：{str(message.content).strip()}")
        elif isinstance(message, SystemMessage):
            lines.append(f"系统：{str(message.content).strip()}")

    return "\n".join(line for line in lines if line.strip())


def _messages_for_direct_answer(state: AdaptiveRagState) -> list[BaseMessage]:
    # Reuse existing dialogue turns so direct-answer mode remains conversational.
    turns: list[BaseMessage] = [
        message for message in state["messages"] if not isinstance(message, SystemMessage)
    ]
    if not turns:
        turns = [HumanMessage(content=_current_question(state))]
    return turns


def _messages_for_agent(state: AdaptiveRagState) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in state["messages"]:
        content = str(message.content).strip()
        if not content:
            continue
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "content": content})
        elif isinstance(message, AIMessage):
            messages.append({"role": "assistant", "content": content})
    return messages


def _extract_final_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text
            elif isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                if parts:
                    return "\n".join(parts)
    return ""


def _build_context_blocks(docs_with_scores: list[tuple[Any, float]]) -> str:
    contexts: list[str] = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", -1)
        contexts.append(
            f"[{idx}] source={source}, chunk={chunk_idx}, score={score:.4f}\n{doc.page_content}"
        )
    return "\n\n".join(contexts)


async def _answer_with_context(
    *,
    question: str,
    rewritten_question: str,
    contexts: str,
    dialogue_context: str,
) -> str:
    model = get_chat_model()
    result = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "你是企业知识库问答助手。"
                    "只能根据给定上下文回答，禁止编造。"
                    "若证据不足，请明确说明“依据不足”。"
                    "答案请精炼，并在末尾给出引用编号，如 [1][2]。"
                )
            ),
            HumanMessage(
                content=(
                    f"多轮对话：\n{dialogue_context}\n\n"
                    f"原始问题：{question}\n"
                    f"改写检索问题：{rewritten_question}\n\n"
                    "可用上下文如下：\n"
                    f"{contexts}"
                )
            ),
        ]
    )
    return str(result.content)


def _retrieve(
    *,
    query: str,
    collection_name: str | None,
    knowledge_domain: str | None,
    book_id: str | None,
    top_k: int | None,
) -> tuple[list[tuple[Any, float]], list[dict[str, Any]]]:
    selected_collection = collection_name or RAG_CHROMA_COLLECTION
    selected_top_k = top_k or RAG_RETRIEVAL_TOP_K
    selected_knowledge_domain = (knowledge_domain or "").strip()
    selected_book_id = (book_id or "").strip()

    metadata_filter: dict[str, Any] | None = None
    filter_map: dict[str, Any] = {}
    if selected_knowledge_domain:
        filter_map["domain"] = selected_knowledge_domain
    if selected_book_id:
        filter_map["book_id"] = selected_book_id
    if filter_map:
        metadata_filter = filter_map

    docs_with_scores = search_chroma(
        query=query.strip(),
        top_k=selected_top_k,
        collection_name=selected_collection,
        metadata_filter=metadata_filter,
    )
    return docs_with_scores, build_citations(docs_with_scores)


async def route_decision_node(state: AdaptiveRagState) -> dict[str, Any]:
    question = _current_question(state)
    dialogue_context = _conversation_preview(state)
    model = get_chat_model().with_structured_output(RouteDecision)
    decision_raw = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_ROUTER_PROMPT),
            HumanMessage(
                content=(
                    "请基于对话上下文判断路由策略。\n"
                    f"对话上下文：\n{dialogue_context}\n\n"
                    f"当前问题：{question}"
                )
            ),
        ]
    )

    if isinstance(decision_raw, RouteDecision):
        decision = decision_raw
    elif isinstance(decision_raw, BaseModel):
        decision = RouteDecision.model_validate(decision_raw.model_dump())
    elif isinstance(decision_raw, dict):
        decision = RouteDecision.model_validate(decision_raw)
    else:
        raise InvalidRequestError(message="LLM 路由结果格式非法", detail={"type": str(type(decision_raw))})

    route = str(decision.route)
    if route not in {
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    }:
        raise InvalidRequestError(message="LLM 路由结果非法", detail={"route": route})

    return {
        "route": route,
        "route_reason": decision.reason.strip() or "无",
    }


async def direct_answer_node(state: AdaptiveRagState) -> dict[str, Any]:
    model = get_chat_model()
    result = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_DIRECT_PROMPT),
            *_messages_for_direct_answer(state),
        ]
    )
    reason = _extract_route_reason(state)
    answer = str(result.content)
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "citations": [],
        "retrieval_count": 0,
        "trace": _build_trace(route="direct_answer", reason=reason, retrieval_count=0),
    }


async def fixed_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    question = _current_question(state)
    dialogue_context = _conversation_preview(state)
    docs_with_scores, citations = _retrieve(
        query=question,
        collection_name=state.get("collection_name"),
        knowledge_domain=state.get("knowledge_domain"),
        book_id=state.get("book_id"),
        top_k=state.get("top_k"),
    )

    if not docs_with_scores:
        answer = "没有检索到相关资料，当前无法基于知识库给出可靠答案。"
    else:
        contexts = _build_context_blocks(docs_with_scores)
        answer = await _answer_with_context(
            question=question,
            rewritten_question=question,
            contexts=contexts,
            dialogue_context=dialogue_context,
        )

    reason = _extract_route_reason(state)
    retrieval_count = 1
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "citations": citations,
        "retrieval_count": retrieval_count,
        "trace": _build_trace(route="fixed_rag", reason=reason, retrieval_count=retrieval_count),
    }


async def _rewrite_query(question: str) -> str:
    model = get_chat_model().with_structured_output(QueryRewrite)
    rewrite_raw = await model.ainvoke(
        [
            SystemMessage(content=ADAPTIVE_RAG_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
    )

    if isinstance(rewrite_raw, QueryRewrite):
        rewrite_result = rewrite_raw
    elif isinstance(rewrite_raw, BaseModel):
        rewrite_result = QueryRewrite.model_validate(rewrite_raw.model_dump())
    elif isinstance(rewrite_raw, dict):
        rewrite_result = QueryRewrite.model_validate(cast(dict[str, Any], rewrite_raw))
    else:
        raise InvalidRequestError(message="LLM 改写结果格式非法", detail={"type": str(type(rewrite_raw))})

    rewritten = rewrite_result.rewritten_query.strip()
    if not rewritten:
        rewritten = question.strip()
    return rewritten


async def agent_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    reason = _extract_route_reason(state)
    question = _current_question(state)
    dialogue_context = _conversation_preview(state)
    knowledge_domain = str(state.get("knowledge_domain") or "").strip()
    domain_text = knowledge_domain or "未指定领域"

    runtime: dict[str, Any] = {
        "rewritten_question": question,
        "docs_with_scores": [],
        "citations": [],
        "retrieval_count": 0,
    }

    @tool
    async def rewrite_query(query: str) -> str:
        """Rewrite user query into a concise retrieval-friendly query."""
        rewritten = await _rewrite_query(query)
        runtime["rewritten_question"] = rewritten
        return rewritten

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant knowledge-base chunks for a query and return formatted context."""
        docs_with_scores, citations = _retrieve(
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
        return _build_context_blocks(docs_with_scores)

    model = get_chat_model()
    agent = create_agent(model=model, tools=[rewrite_query, retrieve_context])
    agent_messages = _messages_for_agent(state)
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
    answer = _extract_final_text(final_messages)
    docs_with_scores = runtime["docs_with_scores"]
    citations = runtime["citations"]
    retrieval_count = int(runtime["retrieval_count"])
    rewritten_question = str(runtime["rewritten_question"])

    if not answer:
        if not docs_with_scores:
            answer = "没有检索到相关资料，当前无法基于知识库给出可靠答案。"
        else:
            contexts = _build_context_blocks(docs_with_scores)
            answer = await _answer_with_context(
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
        "trace": _build_trace(
            route="agent_rag",
            reason=reason,
            retrieval_count=retrieval_count,
        ),
    }


async def strict_insufficient_node(state: AdaptiveRagState) -> dict[str, Any]:
    reason = _extract_route_reason(state)
    answer = "依据不足，当前无法给出可靠答案。"
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "citations": [],
        "retrieval_count": 0,
        "trace": _build_trace(route="strict_insufficient", reason=reason, retrieval_count=0),
    }


def route_selector(state: AdaptiveRagState) -> RagRoute:
    route = state.get("route")
    if route in {
        "direct_answer",
        "fixed_rag",
        "agent_rag",
        "strict_insufficient",
    }:
        return route
    raise InvalidRequestError(message="缺少有效路由结果", detail={"route": route})
