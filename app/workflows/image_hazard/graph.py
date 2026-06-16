from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.workflows.image_hazard.nodes import llm_hazard_detect_node, sam_detect_node
from app.workflows.image_hazard.state import ImageHazardState


def build_image_hazard_graph():
    graph_builder = StateGraph(ImageHazardState)
    graph_builder.add_node("llm_hazard_detect", llm_hazard_detect_node)
    graph_builder.add_node("sam_detect", sam_detect_node)

    graph_builder.add_edge(START, "llm_hazard_detect")
    graph_builder.add_edge("llm_hazard_detect", "sam_detect")
    graph_builder.add_edge("sam_detect", END)

    return graph_builder.compile()
