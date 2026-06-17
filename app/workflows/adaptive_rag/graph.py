from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.workflows.adaptive_rag.nodes.agent_rag import agent_rag_node
from app.workflows.adaptive_rag.nodes.route import route_decision_node, route_selector
from app.workflows.adaptive_rag.state import AdaptiveRagState

_adaptive_rag_graph: Any | None = None
_adaptive_rag_checkpointer = InMemorySaver()


def build_adaptive_rag_graph():
    global _adaptive_rag_graph
    if _adaptive_rag_graph is not None:
        return _adaptive_rag_graph

    g = StateGraph(AdaptiveRagState)
    g.add_node("route_decision", route_decision_node)
    g.add_node("agent_rag", agent_rag_node)

    g.add_edge(START, "route_decision")
    g.add_conditional_edges("route_decision", route_selector, {"agent_rag": "agent_rag"})
    g.add_edge("agent_rag", END)

    _adaptive_rag_graph = g.compile(checkpointer=_adaptive_rag_checkpointer)
    return _adaptive_rag_graph