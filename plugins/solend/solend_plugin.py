from typing import List, Dict
import requests

from plugins.plugin import Plugin
from plugins.types import Pool, Token

class SolendPlugin(Plugin):

    def initialize(self):
        pass

    def fetch_pools(self) -> List[Pool]:
        return []
