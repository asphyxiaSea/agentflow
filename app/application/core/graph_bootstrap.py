from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.application.workflows.adaptive_rag.graph import build_graph_structure


async def bootstrap_rag_graph(redis_url: str):
    """在应用启动阶段调用一次，完成异步初始化，返回图对象和 saver（saver 用于关闭时清理连接）。
    具体构造方式需要对照你安装的 langgraph-checkpoint-redis 版本核实：
    有的版本是 AsyncRedisSaver(redis_url=...)，有的需要 AsyncRedisSaver.from_conn_string(...)。
    """
    saver = AsyncRedisSaver(redis_url=redis_url)
    await saver.asetup()  # 如果这个方法不存在，说明不需要这一步，删掉即可
    graph = build_graph_structure(saver)
    return graph, saver