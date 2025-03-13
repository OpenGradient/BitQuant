from typing import List, Dict
import json

from defillama import DefiLlama

from api.api_types import Pool, Chain, PoolQuery, Token, PoolType


class DefiLlamaProtocols:
    llama: DefiLlama
    pools: List[Pool]
    tokenlist: Dict

    def __init__(self):
        self.llama = DefiLlama()
        self.pools = []

        with open("static/tokenlist.json", "r") as f:
            self.tokenlist = json.load(f)

    def get_pools(self, query: PoolQuery) -> List[Pool]:
        filtered_pools = self.pools.copy()

        if query.chain is not None:
            filtered_pools = [
                pool for pool in filtered_pools if pool.chain == query.chain
            ]

        if query.tokens:
            filtered_pools = [
                pool
                for pool in filtered_pools
                if any(
                    token in [t.address for t in pool.tokens] for token in query.tokens
                )
            ]

        if query.protocols is not None and len(query.protocols) > 0:
            filtered_pools = [
                pool for pool in filtered_pools if pool.protocol in query.protocols
            ]

        if query.isStableCoin is not None:
            filtered_pools = [
                pool
                for pool in filtered_pools
                if pool.isStableCoin == query.isStableCoin
            ]

        if query.impermanentLossRisk is not None:
            filtered_pools = [
                pool
                for pool in filtered_pools
                if pool.impermanentLossRisk == query.impermanentLossRisk
            ]

        return filtered_pools

    def refresh_metrics(self):
        pools_response = self.llama.get_pools()
        self.pools = [self._convert_to_pool(p) for p in pools_response["data"]]

    def _convert_to_pool(self, pool_data: Dict) -> Pool:
        return Pool(
            id=pool_data["pool"],
            chain=DefiLlamaProtocols._get_chain(pool_data["chain"]),
            tokens=[
                Token(
                    address=token_address,
                    symbol=(
                        self.tokenlist[token_address]["symbol"]
                        if token_address in self.tokenlist
                        else ""
                    ),
                    name=(
                        self.tokenlist[token_address]["name"]
                        if token_address in self.tokenlist
                        else ""
                    ),
                )
                for token_address in (pool_data.get("underlyingTokens") or [])
                if token_address is not None
            ],
            TVL=f"${pool_data['tvlUsd']}",
            APRLastDay=pool_data["apy"],
            APRLastWeek=pool_data["apyMean30d"],  # use 30d
            APRLastMonth=pool_data["apyMean30d"],
            protocol=pool_data["project"],
            isStableCoin=pool_data["stablecoin"],
            impermanentLossRisk=pool_data["ilRisk"],
            type=PoolType.LENDING,
            risk="Low",
        )

    @staticmethod
    def _get_chain(chain: str) -> Chain:
        if chain == "Ethereum":
            return Chain.ETHEREUM
        if chain == "Solana":
            return Chain.SOLANA
        if chain == "Base":
            return Chain.BASE
        else:
            return Chain.OTHER
