from typing import List

from api.api_types import Pool
from defi.pools.protocol import Protocol


# https://api.orca.so/docs#/operations/listPools
class OrcaProtocol(Protocol):

    def get_pools(self) -> List[Pool]:
        return []
