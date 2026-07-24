from contextlib import AsyncExitStack

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.domain.workflows.adaptive_rag.graph import build_graph_structure
from app.domain.workflows.pdf_structured.graph import build_pdf_structured_graph


async def bootstrap_rag_graph(redis_url: str, exit_stack: AsyncExitStack):
    """经 langgraph-checkpoint-redis==0.5.1 源码核实:
    AsyncRedisSaver 必须通过 async with 进入/退出生命周期才能正确清理连接
    (它没有独立的 aclose() 方法,清理逻辑封装在 __aexit__ 里)。
    用 AsyncExitStack 把它的生命周期和 FastAPI app 的 lifespan 绑定。
    """
    saver = await exit_stack.enter_async_context(AsyncRedisSaver(redis_url=redis_url))
    graph = build_graph_structure(saver)
    return graph, saver


async def bootstrap_pdf_graph(redis_url: str):
    _ = redis_url
    return build_pdf_structured_graph()