from __future__ import annotations
from functools import lru_cache

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.workflows.adaptive_rag.nodes.agent_rag import agent_rag_node
from app.workflows.adaptive_rag.nodes.route import route_decision_node, route_selector
from app.workflows.adaptive_rag.state import AdaptiveRagState

@lru_cache(maxsize=1)
def build_adaptive_rag_graph():
    g = StateGraph(AdaptiveRagState)
    g.add_node("route_decision", route_decision_node)
    g.add_node("agent_rag", agent_rag_node)

    g.add_edge(START, "route_decision")
    g.add_conditional_edges("route_decision", route_selector, {"agent_rag": "agent_rag"})
    g.add_edge("agent_rag", END)

    return g.compile(checkpointer=InMemorySaver())