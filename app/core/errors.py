from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    message: str = "Internal server error"

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: Any | None = None,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message or self.message)
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        self.detail = detail


class InvalidRequestError(AppError):
    status_code = 400
    code = "invalid_request"
    message = "请求参数不合法"


class PermissionDeniedError(AppError):
    status_code = 403
    code = "permission_denied"
    message = "无权访问该资源"


class ExternalServiceError(AppError):
    status_code = 502
    code = "external_service_error"
    message = "外部服务异常"


class QueueFullError(AppError):
    status_code = 503
    code = "queue_full"
    message = "任务队列已满，请稍后重试"


class SessionNotFoundError(AppError):
    status_code = 404
    code = "session_not_found"
    message = "会话不存在或状态已失效"


class SessionConflictError(AppError):
    status_code = 409
    code = "session_conflict"
    message = "会话已有进行中的任务"
