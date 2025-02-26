from typing import List, Dict

from defillama import DefiLlama

from plugins.types import Pool, Chain, PoolQuery


class DeFiMetrics:

    llama: DefiLlama
    pools: List[Pool]

    def __init__(self):
        self.llama = DefiLlama()

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
                if all(token in pool.tokens for token in query.tokens)
            ]

        if query.protocol is not None:
            filtered_pools = [
                pool for pool in filtered_pools if pool.protocol == query.protocol
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

    def refresh_metrics(self) -> List[Pool]:
        pools_response = self.llama.get_pools()

        return [DeFiMetrics._convert_to_pool(p) for p in pools_response["data"]]

    @staticmethod
    def _convert_to_pool(pool_data: Dict) -> Pool:
        return Pool(
            id=pool_data["pool"],
            chain=DeFiMetrics._get_chain(pool_data["chain"]),
            # tokens=[Token(symbol=token, price=0) for token in pool_data['underlyingTokens']],
            tokens=[],
            TVL=f"${pool_data['tvlUsd']}",
            APRLastDay=pool_data["apy"],
            APRLastWeek=pool_data["apyMean30d"],  # use 30d
            APRLastMonth=pool_data["apyMean30d"],
            protocol=pool_data["project"],
            isStableCoin=pool_data["stablecoin"],
            impermanentLossRisk=pool_data["ilRisk"],
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
