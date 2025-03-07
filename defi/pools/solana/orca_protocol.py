from typing import List

from api.types import Pool
from defi.pools import Protocol

# https://api.orca.so/docs#/operations/listPools
class OrcaProtocol(Protocol):

    def get_pools() -> List[Pool]:
        pass
