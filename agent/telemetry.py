from datadog import statsd
import os
from functools import wraps
import time
import logging


def track_tool_usage(tool_name: str):
    """Track tool usage metrics in Datadog"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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


        return wrapper

    return decorator
