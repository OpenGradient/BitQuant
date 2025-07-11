from datadog import statsd
import os
from functools import wraps
import time
import logging
import inspect


def track_tool_usage(tool_name: str):
    """Track tool usage metrics in Datadog"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Track successful tool usage
                tags = [
                    f"tool_name:{tool_name}",
                    f"environment:{os.environ.get('ENVIRONMENT', 'development')}",
                ]
                statsd.increment("tool.usage.count", tags=tags)
                statsd.histogram("tool.execution.duration", duration, tags=tags)

                return result
            except Exception:
                duration = time.time() - start_time

                # Track failed tool usage
                tags = [
                    f"tool_name:{tool_name}",
                ]
                statsd.increment("tool.errors.count", tags=tags)
                statsd.histogram("tool.execution.duration", duration, tags=tags)

                logging.exception(
                    f"Error in tool: {tool_name} with input {args} and kwargs {kwargs}"
                )
                return f"ERROR: Failed to execute tool {tool_name}."

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Track successful tool usage
                tags = [
                    f"tool_name:{tool_name}",
                    f"environment:{os.environ.get('ENVIRONMENT', 'development')}",
                ]
                statsd.increment("tool.usage.count", tags=tags)
                statsd.histogram("tool.execution.duration", duration, tags=tags)

                return result
            except Exception as e:
                duration = time.time() - start_time

                # Track failed tool usage
                tags = [
                    f"tool_name:{tool_name}",
                    f"error_type:{type(e).__name__}",
                ]
                statsd.increment("tool.errors.count", tags=tags)
                statsd.histogram("tool.execution.duration", duration, tags=tags)

                logging.error(
                    f"Error in tool: {tool_name} with input {args} and kwargs {kwargs}: {e}"
                )
                return f"ERROR: Failed to execute tool {tool_name}."

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
