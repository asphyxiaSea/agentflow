from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from time import time
from typing import Any, Awaitable, Callable, Literal, cast
from uuid import uuid4

from app.core.errors import InvalidRequestError, QueueFullError, TaskNotFoundError
from app.core.settings import (
    TASK_CLEANUP_INTERVAL_SECONDS,
    TASK_QUEUE_MAXSIZE,
    TASK_RESULT_TTL_SECONDS,
    TASK_TIMEOUT_SECONDS,
    TASK_WORKER_COUNT,
)

TaskStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED", "INTERRUPTED"]


class TaskType(StrEnum):
    RAG_CHAT = "rag_chat"
    RAG_CHAT_RESUME = "rag_chat_resume"
    PDF_STRUCTURED = "pdf_structured"


@dataclass
class TaskRecord:
    task_id: str
    task_type: TaskType
    status: TaskStatus
    payload: dict[str, Any]
    created_at: float
    updated_at: float
    result: dict[str, Any] | None = None
    error: str | None = None
    interrupt_payload: dict[str, Any] | None = None   # interrupt 时的 tool_calls

# 简单任务的 handler 签名：只接收业务 payload
SimpleTaskHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
# 支持 interrupt 的 handler 签名：额外接收 task_id 和 dispatcher，用于标记中断状态
InterruptTaskHandler = Callable[[dict[str, Any], str, "TaskDispatcherService"], Awaitable[dict[str, Any]]]

TaskHandler = SimpleTaskHandler | InterruptTaskHandler


class TaskDispatcherService:
    def __init__(
        self,
        *,
        queue_maxsize: int,
        worker_count: int,
        task_timeout_seconds: float,
        result_ttl_seconds: int,
        cleanup_interval_seconds: int,
    ) -> None:
        self._queue: asyncio.Queue[tuple[str, TaskType, dict[str, Any]]] = asyncio.Queue(
            maxsize=max(queue_maxsize, 1)
        )
        self._worker_count = max(worker_count, 1)
        self._task_timeout_seconds = max(task_timeout_seconds, 1.0)
        self._result_ttl_seconds = max(result_ttl_seconds, 60)
        self._cleanup_interval_seconds = max(cleanup_interval_seconds, 10)

        self._handlers: dict[TaskType, tuple[TaskHandler, bool]] = {}
        self._tasks: dict[str, TaskRecord] = {}
        self._task_lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._cleanup_task: asyncio.Task[None] | None = None
        self._started = False
        self._logger = logging.getLogger(__name__)

    def register_handler(self, task_type: TaskType, handler: Callable, supports_interrupt: bool = False) -> None:
        self._handlers[task_type] = (handler, supports_interrupt)

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._workers = [
            asyncio.create_task(
                self._worker_loop(worker_index),
                name=f"dispatcher-worker-{worker_index}",
            )
            for worker_index in range(self._worker_count)
        ]
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="dispatcher-cleanup")
        self._logger.info(
            "Task dispatcher started: workers=%s, queue_maxsize=%s",
            self._worker_count,
            self._queue.maxsize,
        )

    async def stop(self) -> None:
        if not self._started:
            return

        self._started = False
        for worker in self._workers:
            worker.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._cleanup_task:
            await asyncio.gather(self._cleanup_task, return_exceptions=True)

        self._workers = []
        self._cleanup_task = None
        self._logger.info("Task dispatcher stopped")

    async def submit_task(self, *, task_type: TaskType, payload: dict[str, Any]) -> str:
        if not self._started:
            raise RuntimeError("Task dispatcher is not started")

        if task_type not in self._handlers:
            raise InvalidRequestError(message="未注册的任务类型", detail=str(task_type))

        if self._queue.full():
            raise QueueFullError()

        task_id = uuid4().hex
        now = time()
        record = TaskRecord(
            task_id=task_id,
            task_type=task_type,
            status="PENDING",
            payload=payload,
            created_at=now,
            updated_at=now,
        )

        async with self._task_lock:
            self._tasks[task_id] = record

        try:
            self._queue.put_nowait((task_id, task_type, payload))
        except asyncio.QueueFull as exc:
            async with self._task_lock:
                self._tasks.pop(task_id, None)
            raise QueueFullError() from exc

        return task_id

    async def _mark_interrupted(self, task_id: str, interrupt_payload: dict[str, Any]) -> None:
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "INTERRUPTED"
                task.interrupt_payload = interrupt_payload
                task.updated_at = time()

    async def resume_task(self, task_id: str, payload: dict[str, Any]) -> None:
        """复用旧 TaskRecord：从旧 payload 里取出 session_id 合并进新 payload，
        状态改回 PENDING，重新塞进队列触发某个空闲 worker 执行
        RAG_CHAT_RESUME 类型的 handler。
        TaskRecord.task_type 不变（始终代表这条记录最初的任务类型），
        实际执行哪个 handler 由入队元组里单独传入的 task_type 决定。
        """
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                raise TaskNotFoundError()
            if task.status != "INTERRUPTED":
                raise InvalidRequestError(message="任务不在等待确认状态", detail=task.status)

            old_session_id = task.payload.get("session_id")
            if not old_session_id:
                raise InvalidRequestError(message="任务缺少 session_id，无法恢复")

            resume_payload = {**payload, "session_id": old_session_id}
            task.status = "PENDING"
            task.payload = resume_payload
            task.interrupt_payload = None
            task.updated_at = time()

        try:
            self._queue.put_nowait((task_id, TaskType.RAG_CHAT_RESUME, resume_payload))
        except asyncio.QueueFull as exc:
            async with self._task_lock:
                task.status = "INTERRUPTED"  # 入队失败，状态回滚，避免悬空在 PENDING
                task.updated_at = time()
            raise QueueFullError() from exc

    async def get_task_snapshot(self, task_id: str) -> dict[str, Any]:
        async with self._task_lock:
            task = self._tasks.get(task_id)

        if not task:
            raise TaskNotFoundError()

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "created_at": self._format_timestamp(task.created_at),
            "updated_at": self._format_timestamp(task.updated_at),
            "result": task.result,
            "error": task.error,
            "interrupt_payload": task.interrupt_payload,
        }

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _worker_loop(self, worker_index: int) -> None:
        while True:
            task_id, task_type, payload = await self._queue.get()
            try:
                await self._mark_running(task_id)

                started_at = time()
                result = await asyncio.wait_for(
                    self._dispatch(task_type, payload, task_id),
                    timeout=self._task_timeout_seconds,
                )

                if isinstance(result, dict) and result.get("__pending_interrupt__"):
                    # handler 内部已经调用过 _mark_interrupted，这里不再标记 SUCCESS
                    self._logger.info(
                        "Task interrupted: task_id=%s task_type=%s worker=%s queue_size=%s",
                        task_id,
                        task_type,
                        worker_index,
                        self._queue.qsize(),
                    )
                else:
                    await self._mark_success(task_id, result)
                    cost_ms = int((time() - started_at) * 1000)
                    self._logger.info(
                        "Task success: task_id=%s task_type=%s worker=%s cost_ms=%s queue_size=%s",
                        task_id,
                        task_type,
                        worker_index,
                        cost_ms,
                        self._queue.qsize(),
                    )
            except asyncio.TimeoutError:
                await self._mark_failed(task_id, "任务执行超时")
                self._logger.warning(
                    "Task timeout: task_id=%s task_type=%s worker=%s queue_size=%s",
                    task_id,
                    task_type,
                    worker_index,
                    self._queue.qsize(),
                )
            except Exception as exc:  # pragma: no cover
                await self._mark_failed(task_id, str(exc))
                self._logger.exception(
                    "Task failed: task_id=%s task_type=%s worker=%s queue_size=%s",
                    task_id,
                    task_type,
                    worker_index,
                    self._queue.qsize(),
                )
            finally:
                self._queue.task_done()

    async def _dispatch(self, task_type: TaskType, payload: dict[str, Any], task_id: str) -> dict[str, Any]:
        entry = self._handlers.get(task_type)
        if not entry:
            raise InvalidRequestError(message="未注册的任务处理器", detail=str(task_type))
        handler, supports_interrupt = entry
        if supports_interrupt:
            return await cast(InterruptTaskHandler, handler)(payload, task_id, self)
        return await cast(SimpleTaskHandler, handler)(payload)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval_seconds)
            now = time()
            async with self._task_lock:
                expired_ids = [
                    task_id
                    for task_id, task in self._tasks.items()
                    if task.status in ("SUCCESS", "FAILED")
                    and now - task.updated_at > self._result_ttl_seconds
                ]
                for task_id in expired_ids:
                    self._tasks.pop(task_id, None)
            if expired_ids:
                self._logger.info("Cleaned expired tasks: count=%s", len(expired_ids))

    async def _mark_running(self, task_id: str) -> None:
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "RUNNING"
                task.updated_at = time()

    async def _mark_success(self, task_id: str, result: dict[str, Any]) -> None:
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "SUCCESS"
                task.result = result
                task.error = None
                task.updated_at = time()

    async def _mark_failed(self, task_id: str, error: str) -> None:
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "FAILED"
                task.result = None
                task.error = error
                task.updated_at = time()

    @staticmethod
    def _format_timestamp(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


_task_dispatcher_service = TaskDispatcherService(
    queue_maxsize=TASK_QUEUE_MAXSIZE,
    worker_count=TASK_WORKER_COUNT,
    task_timeout_seconds=TASK_TIMEOUT_SECONDS,
    result_ttl_seconds=TASK_RESULT_TTL_SECONDS,
    cleanup_interval_seconds=TASK_CLEANUP_INTERVAL_SECONDS,
)


def get_task_dispatcher_service() -> TaskDispatcherService:
    return _task_dispatcher_service