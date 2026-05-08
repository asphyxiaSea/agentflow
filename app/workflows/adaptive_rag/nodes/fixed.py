from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.workflows.adaptive_rag.nodes.common import (
    answer_with_context,
    build_context_blocks,
    build_trace,
    conversation_preview,
    current_question,
    extract_route_reason,
    retrieve,
)
from app.workflows.adaptive_rag.state import AdaptiveRagState


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
