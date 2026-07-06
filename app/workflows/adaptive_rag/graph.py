from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.workflows.adaptive_rag.nodes.rag_agent.rag_llm_node import (
    finalize_node,
    llm_call_node,
)
from app.workflows.adaptive_rag.nodes.rag_agent.rag_tool_node import rag_tool_node
from app.workflows.adaptive_rag.nodes.route_decision_node import route_decision_node, route_selector
from app.workflows.adaptive_rag.state import AdaptiveRagState
from app.workflows.adaptive_rag.nodes.direct_answer_node import direct_answer_node
from app.workflows.adaptive_rag.nodes.rag_agent.rag_interrupt_node import interrupt_node

@lru_cache(maxsize=1)
def build_adaptive_rag_graph():
    g = StateGraph(AdaptiveRagState)

    g.add_node("route_decision", route_decision_node)
    g.add_node("llm_call", llm_call_node)
    g.add_node("interrupt", interrupt_node)
    g.add_node("tools", rag_tool_node)
    g.add_node("finalize", finalize_node)
    g.add_node("direct_answer", direct_answer_node)

    g.add_edge(START, "route_decision")
    g.add_conditional_edges(
        "route_decision",
        route_selector,
        {
            "llm_call": "llm_call",
            "direct_answer": "direct_answer",  # 新增
        },
    )
    g.add_edge("llm_call", "interrupt")
    
    g.add_conditional_edges(
        "interrupt",
        tools_condition,
        {"tools": "tools", "__end__": "finalize"},
    )
    g.add_edge("tools", "llm_call")
    g.add_edge("finalize", END)
    g.add_edge("direct_answer", END)  # 新增

    return g.compile(checkpointer=AsyncRedisSaver())