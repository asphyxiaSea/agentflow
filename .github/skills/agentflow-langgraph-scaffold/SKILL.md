---
name: agentflow-langgraph-scaffold
description: '用于搭建 LangGraph 框架骨架（非业务链路实现）。适用于从零建立 state、node、graph、pipeline、router 分层结构，链路细节后续填充。关键词：langgraph、框架搭建、骨架、scaffold、bootstrap。'
argument-hint: '例如：先搭一个可运行的 LangGraph 框架骨架，节点逻辑留空'
user-invocable: true
---

# LangGraph 框架搭建技能

## 适用场景
- 你想先搭好 LangGraph 框架，再逐步填业务链路。
- 你希望先稳定工程分层和接口契约，避免后续反复重构。
- 你需要一个可复用的 workflow 脚手架，用于快速新增流程图。

## 搭建目标
1. 先定义稳定的数据契约：state、payload、result。
2. 先打通最小可运行闭环：compile、invoke、status、result。
3. 节点先用占位实现，确保图编排可执行。
4. API/Worker 只做调度，不承载业务细节。

## 推荐目录骨架
```text
app/
  workflows/
    <workflow_name>/
      __init__.py
      state.py
      graph.py
      nodes/
        __init__.py
        start_node.py
        route_node.py
        end_node.py
  application/
    pipelines/
      <workflow_name>_pipeline.py
  api/
    router/
      <workflow_name>_router.py
  worker.py
```

## 框架搭建步骤
1. 创建 workflow 目录与 `state.py`、`nodes/`、`graph.py`。
2. 在 state 里只定义最小输入和输出字段。
3. 在 nodes 里先写占位逻辑，返回固定结构。
4. 在 graph 里先连接主干边，再扩展条件分支。
5. 在 pipeline 里统一做入参校验与 graph 调用。
6. 在 router 里暴露 submit/status/result/resume 接口。
7. 在 worker 注册任务函数并绑定 `session_id`。

## 代码骨架模板

### 1) state 模板
```python
from __future__ import annotations

from typing import TypedDict
from typing_extensions import NotRequired


class ExampleState(TypedDict):
    # 必填输入
    user_input: str

    # 可选中间态/输出
    route: NotRequired[str]
    answer: NotRequired[str]
```

### 2) nodes 模板
```python
from __future__ import annotations

from app.workflows.<workflow_name>.state import ExampleState


def start_node(state: ExampleState) -> ExampleState:
    return {"route": "direct"}


def route_node(state: ExampleState) -> ExampleState:
    text = state["user_input"].strip()
    return {"route": "direct" if text else "fallback"}


def end_node(state: ExampleState) -> ExampleState:
    # 占位实现，后续替换为真实链路
    return {"answer": "TODO: fill business logic"}
```

### 3) graph 模板
```python
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.workflows.<workflow_name>.nodes.<node_file> import (
    start_node,
    route_node,
    end_node,
)
from app.workflows.<workflow_name>.state import ExampleState


def _select_next(state: ExampleState) -> str:
    route = state.get("route", "direct")
    if route == "direct":
        return "end_node"
    return "end_node"


def create_example_graph():
    graph = StateGraph(ExampleState)
    graph.add_node("start_node", start_node)
    graph.add_node("route_node", route_node)
    graph.add_node("end_node", end_node)

    graph.add_edge(START, "start_node")
    graph.add_edge("start_node", "route_node")
    graph.add_conditional_edges("route_node", _select_next)
    graph.add_edge("end_node", END)

    return graph.compile()
```

### 4) pipeline 模板
```python
from __future__ import annotations

from typing import Any

from app.workflows.<workflow_name>.graph import create_example_graph


async def run_example_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    graph = create_example_graph()
    state = {"user_input": str(payload.get("user_input", ""))}
    result = await graph.ainvoke(state)
    return {"answer": result.get("answer", "")}
```

### 5) router 模板
```python
from __future__ import annotations

from fastapi import APIRouter, Request

from app.application.pipelines.<workflow_name>_pipeline import run_example_pipeline

router = APIRouter(tags=["<workflow_name>"])


@router.post("/<workflow_name>/run")
async def run_workflow(request: Request):
    payload = await request.json()
    return await run_example_pipeline(payload)
```

## 最小接口规范（建议）
- `POST /.../sessions/{session_id}/chat`: 提交任务
- `GET /.../sessions/{session_id}/status`: 查询状态
- `GET /.../sessions/{session_id}/result`: 获取结果
- `POST /.../sessions/{session_id}/resume`: 中断恢复

## 骨架完成标准
1. graph 能成功 compile。
2. submit -> status -> result 全链路可走通。
3. 节点逻辑即使是占位实现，也能返回结构化结果。
4. 不依赖外部模型服务，也能完成一次 dry-run。

## 后续填充顺序
1. 先替换 `route_node` 决策逻辑。
2. 再替换核心执行节点（LLM/tool/retrieval）。
3. 最后接外部服务（向量库、模型、第三方 API）。
4. 每替换一个节点，保留 mock 回退开关。

## 常见误区
1. 一开始就把业务细节塞进 graph，导致难以维护。
2. state 契约不清，节点之间靠隐式字段通信。
3. router/worker 直接写业务逻辑，破坏分层。
4. 没先跑通骨架，直接接外部依赖导致排错困难。
