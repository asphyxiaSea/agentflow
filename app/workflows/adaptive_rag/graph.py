from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.workflows.adaptive_rag.nodes.llm_call import (
    finalize_node,
    llm_call_node,
)
from app.workflows.adaptive_rag.nodes.tools import rag_tool_node
from app.workflows.adaptive_rag.nodes.route import route_decision_node, route_selector
from app.workflows.adaptive_rag.state import AdaptiveRagState


@lru_cache(maxsize=1)
def build_adaptive_rag_graph():
    g = StateGraph(AdaptiveRagState)

    g.add_node("route_decision", route_decision_node)
    g.add_node("llm_call", llm_call_node)
    g.add_node("tools", rag_tool_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "route_decision")
    g.add_conditional_edges("route_decision", route_selector, {"agent_rag": "llm_call"})

    # tools_condition：ai_msg 有 tool_calls → tools，否则 → finalize
    g.add_conditional_edges(
        "llm_call",
        tools_condition,
        {"tools": "tools", "__end__": "finalize"},
    )
    g.add_edge("tools", "llm_call")   # 工具执行完回到 llm_call
    g.add_edge("finalize", END)

    return g.compile(checkpointer=InMemorySaver())