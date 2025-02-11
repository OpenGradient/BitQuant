from abc import ABC, abstractmethod
from typing import List, Dict

from plugins.types import Pool


class Plugin(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def fetch_pools(self) -> List[Pool]:
        pass
