from typing import Optional, Any
import aioboto3
from contextlib import asynccontextmanager


@asynccontextmanager
async def get_dynamodb_table(
    table_name: str, session: Optional[aioboto3.Session] = None
):
    """
    Context manager for getting a DynamoDB table.

    Args:
        table_name: Name of the DynamoDB table
        session: Optional aioboto3.Session. If not provided, a new session will be created.

    Yields:
        DynamoDB table resource
    """
    if session is None:
        session = aioboto3.Session()

    async with session.resource("dynamodb") as dynamodb:
        table = await dynamodb.Table(table_name)
        yield table


async def get_table(table_name: str, session: Optional[aioboto3.Session] = None):
    """
    Get a DynamoDB table without using a context manager.
    Note: You are responsible for managing the session lifecycle when using this function.

    Args:
        table_name: Name of the DynamoDB table
        session: Optional aioboto3.Session. If not provided, a new session will be created.

    Returns:
        DynamoDB table resource
    """
    if session is None:
        session = aioboto3.Session()

    dynamodb = await session.resource("dynamodb")
    return await dynamodb.Table(table_name)
