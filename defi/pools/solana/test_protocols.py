import unittest
import json

from defi.pools.solana.orca_protocol import OrcaProtocol
from defi.pools.solana.save_protocol import SaveProtocol
from defi.pools.solana.kamino_protocol import KaminoProtocol
from api.api_types import Chain, Pool, PoolQuery


class TestProtocols(unittest.TestCase):
    def test_save(self):
        save = SaveProtocol()
        pools = save.get_pools()

        self.assertGreater(len(pools), 2)
        print([p.model_dump_json() for p in pools])

    def test_orca(self):
        orca = OrcaProtocol()
        pools = orca.get_pools()

        self.assertGreater(len(pools), 2)

    def test_kamino(self):
        kamino = KaminoProtocol()
        pools = kamino.get_pools()

        self.assertEqual(len(pools), 35)
