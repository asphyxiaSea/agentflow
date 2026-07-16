from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.domain.workflows.pdf_structured.nodes import (
    extract_pdf_text_node,
    pdf_preprocess_node,
    structured_output_node,
    text_preprocess_node,
)
from app.domain.workflows.pdf_structured.state import PdfStructuredState


def build_pdf_structured_graph():
    graph_builder = StateGraph(PdfStructuredState)
    graph_builder.add_node("pdf_preprocess", pdf_preprocess_node)
    graph_builder.add_node("extract_pdf_text", extract_pdf_text_node)
    graph_builder.add_node("text_preprocess", text_preprocess_node)
    graph_builder.add_node("structured_output", structured_output_node)

    graph_builder.add_edge(START, "pdf_preprocess")
    graph_builder.add_edge("pdf_preprocess", "extract_pdf_text")
    graph_builder.add_edge("extract_pdf_text", "text_preprocess")
    graph_builder.add_edge("text_preprocess", "structured_output")
    graph_builder.add_edge("structured_output", END)

    return graph_builder.compile()
