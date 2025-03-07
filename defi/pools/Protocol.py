from abc import ABC, abstractmethod
from typing import List

from api.api_types import Pool


class Protocol(ABC):

    @abstractmethod
    def get_pools() -> List[Pool]:
        pass
