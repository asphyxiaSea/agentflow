from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.errors import InvalidRequestError
from app.core.model_factory import get_chat_model
from app.core.settings import RAG_CHROMA_COLLECTION, RAG_RETRIEVAL_TOP_K
from app.infra.clients.chroma_client import build_citations, search_chroma
from app.workflows.adaptive_rag.state import AdaptiveRagState


def build_trace(*, route: str, reason: str, retrieval_count: int) -> dict[str, Any]:
    return {
        "route": route,
        "reason": reason,
        "retrieval_count": retrieval_count,
    }


def extract_route_reason(state: AdaptiveRagState) -> str:
    reason = str(state.get("route_reason") or "")
    return reason.strip() or "路由模型未提供原因"


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


def build_context_blocks(docs_with_scores: list[tuple[Any, float]]) -> str:
    contexts: list[str] = []
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", -1)
        contexts.append(
            f"[{idx}] source={source}, chunk={chunk_idx}, score={score:.4f}\n{doc.page_content}"
        )
    return "\n\n".join(contexts)


async def answer_with_context(
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


def retrieve(
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


async def fixed_rag_node(state: AdaptiveRagState) -> dict[str, Any]:
    question = current_question(state)
    dialogue_context = conversation_preview(state)
    docs_with_scores, citations = retrieve(
        query=question,
        collection_name=state.get("collection_name"),
        knowledge_domain=state.get("knowledge_domain"),
        book_id=state.get("book_id"),
        top_k=state.get("top_k"),
    )

    if not docs_with_scores:
        answer = "没有检索到相关资料，当前无法基于知识库给出可靠答案。"
    else:
        contexts = build_context_blocks(docs_with_scores)
        answer = await answer_with_context(
            question=question,
            rewritten_question=question,
            contexts=contexts,
            dialogue_context=dialogue_context,
        )

    reason = extract_route_reason(state)
    retrieval_count = 1
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "citations": citations,
        "retrieval_count": retrieval_count,
        "trace": build_trace(route="fixed_rag", reason=reason, retrieval_count=retrieval_count),
    }
