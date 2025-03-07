from abc import ABC, abstractmethod
from typing import List

from api.types import Pool

class Plugin(ABC):

    @abstractmethod
    def get_pools() -> List[Pool]:
        pass
