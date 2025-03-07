import unittest

from defi.pools.solana.orca_protocol import OrcaProtocol
from api.api_types import Chain, Pool, PoolQuery


class TestOrcaProtocol(unittest.TestCase):

    def test_orca(self):
        orca = OrcaProtocol()
        pools = orca.get_pools()

        self.assertGreater(len(pools), 2)
        print(pools)
