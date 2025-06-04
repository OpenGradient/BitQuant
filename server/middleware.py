import time
from typing import Awaitable, Callable

from datadog import statsd  # type: ignore[attr-defined]
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class DatadogMetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()

        # Get the path template
        path_template = request.url.path
        for route in request.app.routes:
            if route.path_format == path_template:
                path_template = route.path_format
                break
        else:
            return await call_next(request)

        # Track request count
        statsd.increment(
            "fastapi.request.count",
            tags=[
                f"path:{path_template}",
                f"method:{request.method}",
            ],
        )

        try:
            response = await call_next(request)

            # Track response time and status
            duration = time.time() - start_time
            tags = [
                f"method:{request.method}",
                f"status:{response.status_code}",
                f"path:{path_template if response.status_code != 404 else 'not_found'}",
            ]

            statsd.histogram(
                "fastapi.request.duration",
                duration,
                tags=tags,
            )

            # Track status code distribution
            statsd.increment(
                "fastapi.response.status",
                tags=tags,
            )

            return response

        except Exception as e:
            # Track errors
            statsd.increment(
                "fastapi.request.error",
                tags=[
                    f"path:{path_template}",
                    f"method:{request.method}",
                    f"error:{type(e).__name__}",
                ],
            )
            raise
