import aioboto3
import os


class DatabaseManager:
    def __init__(self):
        self.session: aioboto3.Session = aioboto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION"),
        )

    def get_table_context(self, table_name: str):
        return TableContext(self.session, table_name)

    def table_context_factory(self, table_name: str):
        return lambda: self.get_table_context(table_name)


class TableContext:
    def __init__(self, session, table_name):
        self.session = session
        self.table_name = table_name

    async def __aenter__(self):
        self.dynamodb = self.session.resource("dynamodb")
        self.resource = await self.dynamodb.__aenter__()
        return await self.resource.Table(self.table_name)

    async def __aexit__(self, *args):
        await self.dynamodb.__aexit__(*args)
