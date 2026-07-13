from __future__ import annotations

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
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


def _build_graph_structure(checkpointer: AsyncRedisSaver) -> CompiledStateGraph:
    """纯粹的图结构定义，不涉及连接/资源生命周期，方便复用。"""
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
            "direct_answer": "direct_answer",
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
    g.add_edge("direct_answer", END)

    return g.compile(checkpointer=checkpointer)


async def create_adaptive_rag_graph(redis_url: str) -> tuple[CompiledStateGraph, AsyncRedisSaver]:
    """在应用启动阶段调用一次，完成异步初始化，返回图对象和 saver（saver 用于关闭时清理连接）。
    具体构造方式需要对照你安装的 langgraph-checkpoint-redis 版本核实：
    有的版本是 AsyncRedisSaver(redis_url=...)，有的需要 AsyncRedisSaver.from_conn_string(...)。
    """
    saver = AsyncRedisSaver(redis_url=redis_url)
    await saver.asetup()  # 如果这个方法不存在，说明不需要这一步，删掉即可
    graph = _build_graph_structure(saver)
    return graph, saver