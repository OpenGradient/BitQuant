from typing import List

from .strategy import Strategy
from .max_yield import MaxYieldStrategy

STRATEGIES: List[Strategy] = [
    MaxYieldStrategy(),
]
