from abc import ABC, abstractmethod
from typing import List

from api.api_types import Pool


class Protocol(ABC):

    @abstractmethod
    def get_pools(self) -> List[Pool]:
        pass
