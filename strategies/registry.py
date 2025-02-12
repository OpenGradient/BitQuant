from typing import List, Tuple, Any, Type
from pydantic import BaseModel

from .strategy import Strategy
from .max_yield import MaxYieldStrategy, MaxYieldOptions

STRATEGIES: List[Tuple[Strategy[Any], Type[BaseModel]]] = [
    (MaxYieldStrategy(), MaxYieldOptions)
]
