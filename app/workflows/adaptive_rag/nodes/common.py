from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document

from app.core.errors import InvalidRequestError
from app.workflows.adaptive_rag.state import AdaptiveRagState


def extract_route_reason(state: AdaptiveRagState) -> str:
    return str(state.get("route_reason") or "").strip() or "路由模型未提供原因"


def current_question(state: AdaptiveRagState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and (content := str(msg.content).strip()):
            return content
    raise InvalidRequestError(message="messages 中缺少有效用户问题")


def build_context_blocks(docs_with_scores: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{i}] source={doc.metadata.get('source', 'unknown')}, "
        f"chunk={doc.metadata.get('chunk_index', -1)}, score={score:.4f}\n{doc.page_content}"
        for i, (doc, score) in enumerate(docs_with_scores, 1)
    )